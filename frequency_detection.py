import torch
import torch.nn as nn
import torch.nn.functional as F


def local_average_filter(tensor: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Args:
        tensor: Input Tensor 
                - (H, W)
                - (1, H, W) or (C, H, W) 
                - (1, 1, H, W) or (1, C, H, W)
                - (B, K, C, H, W)
        kernel_size: default number is 3 - This number should be odd number.
    
    Returns:
        Averaged Tensor (Same shape as input tensor)
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    original_shape = tensor.shape
    original_device = tensor.device
    
    # Handle 5D tensors (B, K, C, H, W)
    if tensor.dim() == 5:
        B, K, C, H, W = tensor.shape
        # Reshape to (B*K, C, H, W) for efficient processing
        tensor_reshaped = tensor.view(B * K, C, H, W)
        
        # Apply averaging filter
        padding = kernel_size // 2
        tensor_padded = torch.nn.functional.pad(
            tensor_reshaped, 
            (padding, padding, padding, padding), 
            mode='reflect'
        )
        result = torch.nn.functional.avg_pool2d(
            tensor_padded, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=0
        )
        
        # Reshape back to original dimensions
        result = result.view(B, K, C, H, W)
        return result.to(original_device)
    
    # Handle 2-4D tensors
    if tensor.dim() == 2:  # (H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
    elif tensor.dim() == 3:  # (1, H, W) or (C, H, W)
        if tensor.shape[0] == 1:  # (1, H, W)
            tensor = tensor.unsqueeze(0)  # -> (1, 1, H, W)
        else:  # (C, H, W) - has channel dimension
            tensor = tensor.unsqueeze(0)  # -> (1, C, H, W)
    # Keep (B, C, H, W) tensors as-is
    
    B, C, H, W = tensor.shape
    
    # Create averaging kernel with reflection padding
    padding = kernel_size // 2
    
    # Use avg_pool2d for efficient computation with proper edge handling
    tensor_padded = torch.nn.functional.pad(tensor, (padding, padding, padding, padding), mode='reflect')
    
    # Apply average pooling (stride=1 computes mean at each position)
    result = torch.nn.functional.avg_pool2d(tensor_padded, kernel_size=kernel_size, stride=1, padding=0)
    
    # Restore original tensor shape
    if len(original_shape) == 2:  # (H, W)
        result = result.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:  # (1, H, W) or (C, H, W)
        if original_shape[0] == 1:
            result = result.squeeze(0)
        else:
            result = result.squeeze(0)
    
    return result.to(original_device)



def _is_fastlen(n: int) -> bool:
    """Check if n has only 2, 3, 5, 7 as prime factors (FFT-friendly sizes)."""
    for p in (2, 3, 5, 7):
        while n % p == 0:
            n //= p
    return n == 1


def _next_fastlen(n: int) -> int:
    """Find the smallest FFT-friendly size >= n."""
    m = n
    while not _is_fastlen(m):
        m += 1
    return m

class FFTBandEnergy(nn.Module):
    """
    Efficient frequency domain analysis using rFFT and concurrent band filtering.
    
    Performs a single rFFT on the input image, applies multiple concentric frequency
    band masks, and computes spatial energy maps for each band via irFFT.
    
    Args:
        Input:  (B, 3, H, W) RGB image (assumed in range [0, 1])
        Output: (B, K, H, W) normalized energy maps for each frequency band
    """
    def __init__(self, bands, use_fastlen=True, pad_extra=0, use_local_contrast=True, ksz=17, eps=1e-6):
        """
        Args:
            bands: List of (fmin, fmax) tuples defining normalized frequency bands [0..1]
            use_fastlen: If True, pad to FFT-friendly sizes for cuFFT optimization
            pad_extra: Additional reflection padding in pixels
            use_local_contrast: If True, enhance local contrast using local averaging
            ksz: Kernel size for local averaging
            eps: Small epsilon for numerical stability in normalization
        """
        super().__init__()
        self.bands = list(bands)
        self.use_fastlen = use_fastlen
        self.pad_extra = int(pad_extra)
        self.use_local_contrast = use_local_contrast
        self.ksz = int(ksz)
        self.eps = float(eps)

        # Register buffers for cached frequency grids and band masks
        self.register_buffer('_rr', torch.tensor(0.), persistent=False)      # Normalized frequency radius grid
        self.register_buffer('_masks', torch.tensor(0.), persistent=False)   # Band masks (K, Hp, Wr)
        self._shape_key = None  # Shape key for cache validation

    @torch.no_grad()
    def _to_luma(self, x):
        if x.size(1) == 3:
            r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
            return 0.2126*r + 0.7152*g + 0.0722*b
        return x

    @torch.no_grad()
    def _build_grids_and_masks(self, H: int, W: int, device, dtype):
        """
        Build frequency domain grids and band masks for rFFT (unshifted).
        
        Constructs the normalized frequency radius grid and creates binary masks
        for each frequency band. Results are cached as buffers.
        """
        Wr = W // 2 + 1  # Frequency domain width for rfft2
        # Frequency axes in cycles/pixel
        fy = torch.fft.fftfreq(H, d=1.0, device=device, dtype=dtype)  # [-0.5, +0.5)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device, dtype=dtype)  # [0, +0.5]
        # Normalized radius (Nyquist=0.5 is mapped to 1.0)
        # Max radius is sqrt(0.5^2 + 0.5^2) at (fy=±0.5, fx=0.5)
        yy, xx = torch.meshgrid(fy, fx, indexing='ij')
        rr_raw = torch.sqrt(yy**2 + xx**2)  # Radius in cycles/pixel
        rr = rr_raw / rr_raw.max().clamp_min(1e-12)  # Normalize to [0, 1]
        # Create binary masks for each frequency band
        masks = []
        for (fmin, fmax) in self.bands:
            m = ((rr >= fmin) & (rr < fmax)).to(dtype)  # Shape: (H, Wr)
            masks.append(m)
        masks = torch.stack(masks, dim=0)  # Shape: (K, H, Wr)

        self._rr = rr
        self._masks = masks
        self._shape_key = (H, W)

    @torch.no_grad()
    def forward(self, img_rgb: torch.Tensor):
        assert img_rgb.dim() == 4
        B, C, H, W = img_rgb.shape
        device, dtype = img_rgb.device, img_rgb.dtype

        # Apply reflection padding to FFT-friendly size
        Hp, Wp = H, W
        if self.use_fastlen:
            Hp = _next_fastlen(H)
            Wp = _next_fastlen(W)
        Hp += 2 * self.pad_extra
        Wp += 2 * self.pad_extra
        pad_t = (Hp - H) // 2
        pad_b = Hp - H - pad_t
        pad_l = (Wp - W) // 2
        pad_r = Wp - W - pad_l

        Y = self._to_luma(img_rgb)  # Shape: (B, 1, H, W)
        Yp = F.pad(Y, (pad_l, pad_r, pad_t, pad_b), mode='reflect') if (pad_t or pad_l) else Y
        Hp, Wp = Yp.shape[-2], Yp.shape[-1]
        Wr = Wp // 2 + 1

        # Build or reuse cached frequency grids and masks
        if self._shape_key != (Hp, Wp) or self._masks.numel() == 1:
            self._build_grids_and_masks(Hp, Wp, device, dtype)

        K = len(self.bands)

        # Compute unshifted rFFT
        F_r = torch.fft.rfft2(Yp, dim=(-2, -1))  # Shape: (B, 1, Hp, Wr)

        # Apply band masks and batch compute irFFT
        # Broadcasting: (B, 1, Hp, Wr) × (K, Hp, Wr) -> (B, K, Hp, Wr)
        F_bp = F_r * self._masks[None, None]
        # Reshape for batch irFFT
        F_bp = F_bp.view(B * K, 1, Hp, Wr)
        y_bp = torch.fft.irfft2(F_bp, s=(Hp, Wp), dim=(-2, -1))  # Shape: (B*K, 1, Hp, Wp)
        y_bp = y_bp.view(B, K, Hp, Wp)

        # Crop to original spatial size
        if pad_t or pad_l:
            y_bp = y_bp[..., pad_t:pad_t + H, pad_l:pad_l + W]

        # Compute amplitude and apply local contrast enhancement
        out = y_bp.abs()
        if self.use_local_contrast and self.ksz > 1:
            pad = self.ksz // 2
            blur = F.avg_pool2d(out, kernel_size=self.ksz, stride=1, padding=pad)
            out = torch.relu(out - blur)

        # Normalize to [0, 1] independently per band
        vmin = out.amin(dim=(2, 3), keepdim=True)
        vmax = out.amax(dim=(2, 3), keepdim=True)
        out = (out - vmin) / (vmax - vmin + self.eps)
        return out.clamp_(0, 1)
