/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, 
			const float* error_map,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* W1,
			const float* b1,
			const float* W2,
			const float* b2,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* gaussian_errors,
			float* gaussian_effects,
			float* out_color,
			float* out_others,
			int* radii = nullptr,
			bool debug = false);
	};
};

#endif
