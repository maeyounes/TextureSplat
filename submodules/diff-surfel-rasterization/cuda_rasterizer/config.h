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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// Per-Gaussian splat features (one scalar per channel per primitive).
// When textures are active (TS > 0) only the indirect
// light (3 channels) is routed through splat features.
#define MAX_SPLAT_FEATURES 24
// Per-Gaussian texture features (sampled per-pixel from each primitive's
// texture maps): tangent normal (2/3) + reflection (1) + roughness (1) +
// albedo (3) = up to 8 channels. Zero when textures are disabled.
#define MAX_TEXTURE_FEATURES 8
#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#endif
