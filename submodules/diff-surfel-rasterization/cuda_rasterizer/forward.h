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

// 已修改

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector_types.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "../atlas_view.h"

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* means3D,
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, const int H,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		int* radii,
		float2* means2D,
		float* depths,
		float* transMats,
		float* rgb,
		float4* normal_opacity,
		float3* rotation_matrices,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const int S, const int TS, int W, int H,
		float focal_x, float focal_y,
		const float2* means2D,
		const float* colors,
		const float* splat_features,
		const float* texture_features,
		int texture_size,
		const float* transMats,
		const float* depths,
		const float4* normal_opacity,
		const float3* rotation_matrices,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_feature,
		float* out_others,
		// Hardware bilinear sampling (eval-only). Default = disabled = SW path.
		AtlasView atlas = AtlasView{});
}


#endif
