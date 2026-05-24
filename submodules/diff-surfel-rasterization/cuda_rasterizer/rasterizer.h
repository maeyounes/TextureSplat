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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include "cuda_runtime.h"
#include "../atlas_view.h"

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, const int S, const int TS, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* splat_features,
			const float* texture_features,
			const int texture_size,
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
			int* out_contrib,
			float* out_color,
			float* out_feature,
			float* out_others,
			int* radii = nullptr,
			bool debug = false,
			// Hardware bilinear sampling (eval-only). Default-constructed
			// AtlasView is disabled() and selects the software path.
			AtlasView atlas = AtlasView{});

		static void backward(
			const int P, int S, int TS, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* splat_features,
			const float* texture_features, // //
			const int texture_size,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* img_buffer,
			const int* out_contrib,
			const float* dL_dpix,
			const float* dL_dpix_f, // //
			const float* dL_depths,
			float* dL_dmean2D,
			float* dL_dnormal,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dsplatfeature,
			float* dL_dtexturefeature, // //
			float* dL_dmean3D,
			float* dL_dtransMat,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_drotTexMat,
			bool debug);
	};
};

#endif
