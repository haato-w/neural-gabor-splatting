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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_projector/config.h"
#include "cuda_projector/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

#define CHECK_INPUT(x)											\
	AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
	// AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor>
InverseProjectErrorsCUDA(
	const torch::Tensor& error_map,
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	// const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	// const torch::Tensor& sh,
	const torch::Tensor& W1,
	const torch::Tensor& b1,
	const torch::Tensor& W2,
	const torch::Tensor& b2,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
	AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  if (MLP_OUTPUT_DIM != 3)
  {
	AT_ERROR("MLP_OUTPUT_DIM should be 3");
  }
  if (W1.ndimension() != 3 || W1.size(1) != HIDDEN_NEURON || W1.size(2) != MLP_INPUT_DIM)
  {
	AT_ERROR("W1 must have dimensions (num_points, HIDDEN_NEURON, MLP_INPUT_DIM)");
  }
  if (b1.ndimension() != 2 || b1.size(1) != HIDDEN_NEURON)
  {
	AT_ERROR("b1 must have dimensions (num_points, HIDDEN_NEURON)");
  }
  if (W2.ndimension() != 3 || W2.size(1) != MLP_OUTPUT_DIM || W2.size(2) != HIDDEN_NEURON)
  {
	AT_ERROR("W1 must have dimensions (num_points, MLP_OUTPUT_DIM, HIDDEN_NEURON)");
  }
  if (b2.ndimension() != 2 || b2.size(1) != MLP_OUTPUT_DIM)
  {
	AT_ERROR("b1 must have dimensions (num_points, MLP_OUTPUT_DIM)");
  }

  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  CHECK_INPUT(background);
  CHECK_INPUT(means3D);
  CHECK_INPUT(opacity);
  CHECK_INPUT(scales);
  CHECK_INPUT(rotations);
  CHECK_INPUT(transMat_precomp);
  CHECK_INPUT(viewmatrix);
  CHECK_INPUT(projmatrix);
  CHECK_INPUT(W1);
  CHECK_INPUT(b1);
  CHECK_INPUT(W2);
  CHECK_INPUT(b2);
  CHECK_INPUT(campos);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor gaussian_errors = torch::full({P}, 0.0, float_opts);
  torch::Tensor gaussian_effects = torch::full({P}, 0.0, float_opts);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  rendered = CudaRasterizer::Rasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, 
		// M,
		error_map.contiguous().data<float>(),
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		W1.contiguous().data_ptr<float>(),
		b1.contiguous().data_ptr<float>(),
		W2.contiguous().data_ptr<float>(),
		b2.contiguous().data_ptr<float>(),
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		transMat_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		gaussian_errors.contiguous().data<float>(),
		gaussian_effects.contiguous().data<float>(),
		out_color.contiguous().data<float>(),
		out_others.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, gaussian_errors, gaussian_effects);
}
