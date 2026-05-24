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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Bilinearly sample a per-Gaussian texture map.
//   (u, v): normalized coords in [-1, 1] (align_corners convention).
//   texture: channel-planar, texture[c * size * size + y * size + x].
//   Out-of-bounds corners contribute zero.
// Result is written to result[0 .. channels).
static __forceinline__ __device__
void bilinear_sample(
	float* __restrict__ result,
	float u, float v,
	const float* __restrict__ texture,
	int size,
	int channels)
{
	const float x = (u + 1.0f) * 0.5f * (size - 1);
	const float y = (v + 1.0f) * 0.5f * (size - 1);
	const int ix0 = static_cast<int>(floorf(x));
	const int iy0 = static_cast<int>(floorf(y));
	const int ix1 = ix0 + 1;
	const int iy1 = iy0 + 1;
	const float fx = x - ix0;
	const float fy = y - iy0;

	const float w_nw = (1.0f - fx) * (1.0f - fy);
	const float w_ne = fx          * (1.0f - fy);
	const float w_sw = (1.0f - fx) * fy;
	const float w_se = fx          * fy;

	const bool ok_nw = (ix0 >= 0) && (ix0 < size) && (iy0 >= 0) && (iy0 < size);
	const bool ok_ne = (ix1 >= 0) && (ix1 < size) && (iy0 >= 0) && (iy0 < size);
	const bool ok_sw = (ix0 >= 0) && (ix0 < size) && (iy1 >= 0) && (iy1 < size);
	const bool ok_se = (ix1 >= 0) && (ix1 < size) && (iy1 >= 0) && (iy1 < size);

	const int area = size * size;
	for (int c = 0; c < channels; ++c)
	{
		const float* p = texture + c * area;
		float out = 0.0f;
		if (ok_nw) out += p[iy0 * size + ix0] * w_nw;
		if (ok_ne) out += p[iy0 * size + ix1] * w_ne;
		if (ok_sw) out += p[iy1 * size + ix0] * w_sw;
		if (ok_se) out += p[iy1 * size + ix1] * w_se;
		result[c] = out;
	}
}

// Backward of bilinear_sample. Scatters per-channel grad_out into grad_texture
// via atomicAdd and returns the gradient w.r.t. the (u, v) sampling coords.
static __forceinline__ __device__
float2 bilinear_sample_backward(
	const float* __restrict__ grad_out,         // [channels]
	float u, float v,
	const float* __restrict__ texture,
	float* __restrict__ grad_texture,
	int size,
	int channels)
{
	const float x = (u + 1.0f) * 0.5f * (size - 1);
	const float y = (v + 1.0f) * 0.5f * (size - 1);
	const int ix0 = static_cast<int>(floorf(x));
	const int iy0 = static_cast<int>(floorf(y));
	const int ix1 = ix0 + 1;
	const int iy1 = iy0 + 1;
	const float fx = x - ix0;
	const float fy = y - iy0;

	const float w_nw = (1.0f - fx) * (1.0f - fy);
	const float w_ne = fx          * (1.0f - fy);
	const float w_sw = (1.0f - fx) * fy;
	const float w_se = fx          * fy;

	const bool ok_nw = (ix0 >= 0) && (ix0 < size) && (iy0 >= 0) && (iy0 < size);
	const bool ok_ne = (ix1 >= 0) && (ix1 < size) && (iy0 >= 0) && (iy0 < size);
	const bool ok_sw = (ix0 >= 0) && (ix0 < size) && (iy1 >= 0) && (iy1 < size);
	const bool ok_se = (ix1 >= 0) && (ix1 < size) && (iy1 >= 0) && (iy1 < size);

	const int area = size * size;
	float dL_dx = 0.0f, dL_dy = 0.0f;
	for (int c = 0; c < channels; ++c)
	{
		const float go = grad_out[c];
		const int off = c * area;
		// dL / d(corner): scatter weight * grad_out.
		if (ok_nw) atomicAdd(grad_texture + off + iy0 * size + ix0, w_nw * go);
		if (ok_ne) atomicAdd(grad_texture + off + iy0 * size + ix1, w_ne * go);
		if (ok_sw) atomicAdd(grad_texture + off + iy1 * size + ix0, w_sw * go);
		if (ok_se) atomicAdd(grad_texture + off + iy1 * size + ix1, w_se * go);
		// dL / d(x, y): gather corner_value * d(weight)/d(x, y) * grad_out.
		if (ok_nw)
		{
			const float v_nw = texture[off + iy0 * size + ix0];
			dL_dx -= v_nw * (1.0f - fy) * go;
			dL_dy -= v_nw * (1.0f - fx) * go;
		}
		if (ok_ne)
		{
			const float v_ne = texture[off + iy0 * size + ix1];
			dL_dx += v_ne * (1.0f - fy) * go;
			dL_dy -= v_ne * fx          * go;
		}
		if (ok_sw)
		{
			const float v_sw = texture[off + iy1 * size + ix0];
			dL_dx -= v_sw * fy          * go;
			dL_dy += v_sw * (1.0f - fx) * go;
		}
		if (ok_se)
		{
			const float v_se = texture[off + iy1 * size + ix1];
			dL_dx += v_se * fy          * go;
			dL_dy += v_se * fx          * go;
		}
	}
	// Chain rule: convert from pixel-space d/d(x, y) to normalized d/d(u, v).
	const float jac = (size - 1) * 0.5f;
	return make_float2(dL_dx * jac, dL_dy * jac);
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, const bool *clamped, const glm::vec3 *dL_dcolor, glm::vec3 *dL_dmeans, glm::vec3 *dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3 *dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy +
						   SH_C3[1] * sh[10] * yz +
						   SH_C3[2] * sh[11] * -2.f * xy +
						   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
						   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
						   SH_C3[5] * sh[14] * 2.f * xz +
						   SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (SH_C3[0] * sh[9] * 3.f * (xx - yy) +
						   SH_C3[1] * sh[10] * xz +
						   SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
						   SH_C3[3] * sh[12] * -3.f * 2.f * yz +
						   SH_C3[4] * sh[13] * -2.f * xy +
						   SH_C3[5] * sh[14] * -2.f * yz +
						   SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (SH_C3[1] * sh[10] * xy +
						   SH_C3[2] * sh[11] * 4.f * 2.f * yz +
						   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
						   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
						   SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z}, float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

	// Gradients of loss w.r.t. Gaussian means, but only the portion
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of the rendering procedure.
//
// This kernel serves two modes from one compiled extension:
//  - TS == 0 (baseline): per-Gaussian splat features only, with the 2D
//    screen-space low-pass filter and per-Gaussian normal gradient. The
//    behaviour is bit-for-bit identical to the original 2DGS/Ref-Gaussian
//    rasterizer.
//  - TS  > 0 (textured): per-primitive texture maps are bilinearly sampled
//    at the ray-splat intersection, tangent-space normals are rotated to
//    world space, and gradients flow to the textures, rotation matrices and
//    splat features.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
	renderCUDA(
		const uint2 *__restrict__ ranges,
		const uint32_t *__restrict__ point_list,
		int S, int TS, int W, int H,
		float focal_x, float focal_y,
		const float *__restrict__ bg_color,
		const float2 *__restrict__ points_xy_image,
		const float4 *__restrict__ normal_opacity,
		const float3 *__restrict__ rotation_matrices,
		const float *__restrict__ transMats,
		const float *__restrict__ colors,
		const float *__restrict__ splat_features,	//
		const float *__restrict__ texture_features, //
		int texture_size,
		const float *__restrict__ depths,
		const float *__restrict__ final_Ts,
		const uint32_t *__restrict__ n_contrib,
		const float *__restrict__ dL_dpixels,
		const float *__restrict__ dL_dpixels_f, //
		const float *__restrict__ dL_depths,
		float *__restrict__ dL_dtransMat,
		float3 *__restrict__ dL_dmean2D,
		float *__restrict__ dL_dnormal3D,
		float *__restrict__ dL_dopacity,
		float *__restrict__ dL_dcolors,
		float *__restrict__ dL_dsplat_features,	  //
		float *__restrict__ dL_dtexture_features, //
		float *__restrict__ dL_drotation_matrices //
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix_max = {min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H)};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = {(float)pix.x, (float)pix.y};

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_features[MAX_SPLAT_FEATURES * BLOCK_SIZE]; // splat features (shared by both modes)
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = {0};
	float accum_rec_f[MAX_SPLAT_FEATURES + MAX_TEXTURE_FEATURES] = {0}; // //
	float dL_dpixel[C];
	float dL_dpixel_f[MAX_SPLAT_FEATURES + MAX_TEXTURE_FEATURES];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside)
	{
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++)
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = {0};
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

	if (inside)
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		for (int i = 0; i < S + TS; i++)
			dL_dpixel_f[i] = dL_dpixels_f[i * H * W + pix_id]; // //
	}

	float last_alpha = 0;
	float last_color[C] = {0};
	float last_feature[MAX_SPLAT_FEATURES + MAX_TEXTURE_FEATURES] = {0}; // //

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id + 0], transMats[9 * coll_id + 1], transMats[9 * coll_id + 2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id + 3], transMats[9 * coll_id + 4], transMats[9 * coll_id + 5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id + 6], transMats[9 * coll_id + 7], transMats[9 * coll_id + 8]};
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			// collected_depths[block.thread_rank()] = depths[coll_id];
			for (int i = 0; i < S; i++)
				collected_features[i * BLOCK_SIZE + block.thread_rank()] = splat_features[coll_id * S + i]; // //
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0)
				continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y);
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);

			// compute intersection and depth.
			// Baseline (TS == 0) keeps the 2D screen-space low-pass filter;
			// texture mode (TS > 0) disables it (see forward.cu).
			float rho, c_d;
			if (TS > 0)
			{
				rho = rho3d;
				c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			}
			else
			{
				rho = min(rho3d, rho2d);
				c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
			}
			if (c_d < near_n)
				continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			// accumulations

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, opa * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float w = alpha * T;
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			// Gradient w.r.t. the local intersection coordinate s, fed by
			// texture sampling. Stays zero in the baseline (TS == 0) path.
			float2 dL_ds_textures = {0.0f, 0.0f};

			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian.
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			if (TS > 0)
			{
				// --- Texture mode -------------------------------------------------
				// Re-sample the textures exactly as in the forward pass.
				float sampled_feature[MAX_TEXTURE_FEATURES] = {0};
				const int texture_offset = global_id * texture_size * texture_size * TS;
				const float2 texture_coords = {s.x / 4.0f, s.y / 4.0f};
				bilinear_sample(sampled_feature, texture_coords.x, texture_coords.y,
								texture_features + texture_offset, texture_size, TS);

				// Rotation matrix (tangent -> world) read from global memory.
				float3 rotmat_row0 = rotation_matrices[global_id * 3 + 0];
				float3 rotmat_row1 = rotation_matrices[global_id * 3 + 1];
				float3 rotmat_row2 = rotation_matrices[global_id * 3 + 2];

				// Normalize the sampled tangent-space normal.
				float sampled_normal_unit[3] = {0};
				float norm = sqrt(sampled_feature[0] * sampled_feature[0] +
								  sampled_feature[1] * sampled_feature[1] +
								  sampled_feature[2] * sampled_feature[2]);
				if (norm > 0)
				{
					sampled_normal_unit[0] = sampled_feature[0] / norm;
					sampled_normal_unit[1] = sampled_feature[1] / norm;
					sampled_normal_unit[2] = sampled_feature[2] / norm;
				}

				float world_normal[3] = {0};
				world_normal[0] = rotmat_row0.x * sampled_normal_unit[0] + rotmat_row1.x * sampled_normal_unit[1] + rotmat_row2.x * sampled_normal_unit[2];
				world_normal[1] = rotmat_row0.y * sampled_normal_unit[0] + rotmat_row1.y * sampled_normal_unit[1] + rotmat_row2.y * sampled_normal_unit[2];
				world_normal[2] = rotmat_row0.z * sampled_normal_unit[0] + rotmat_row1.z * sampled_normal_unit[1] + rotmat_row2.z * sampled_normal_unit[2];

				// Splat features are stored after the TS texture channels.
				for (int ch = 0; ch < S; ch++)
				{
					const int off = ch + TS;
					const float splat_feature = collected_features[ch * BLOCK_SIZE + j];
					accum_rec_f[off] = last_alpha * last_feature[off] + (1.f - last_alpha) * accum_rec_f[off];
					last_feature[off] = splat_feature;
					const float dL_dchannel_f = dL_dpixel_f[off];
					dL_dalpha += (splat_feature - accum_rec_f[off]) * dL_dchannel_f;
					atomicAdd(&(dL_dsplat_features[global_id * S + ch]), dchannel_dcolor * dL_dchannel_f);
				}

				// World-space normal occupies texture channels [0, 3).
				float dL_dsampled_feature[MAX_TEXTURE_FEATURES] = {0};
				float dL_dtexture_world_normal[3] = {0};
				for (int ch = 0; ch < 3; ch++)
				{
					accum_rec_f[ch] = last_alpha * last_feature[ch] + (1.f - last_alpha) * accum_rec_f[ch];
					last_feature[ch] = world_normal[ch];
					const float dL_dchannel_f = dL_dpixel_f[ch];
					dL_dalpha += (world_normal[ch] - accum_rec_f[ch]) * dL_dchannel_f;
					dL_dtexture_world_normal[ch] += dchannel_dcolor * dL_dchannel_f;
				}
				// Remaining texture channels [3, TS).
				for (int ch = 3; ch < TS; ch++)
				{
					const float feature = sampled_feature[ch];
					accum_rec_f[ch] = last_alpha * last_feature[ch] + (1.f - last_alpha) * accum_rec_f[ch];
					last_feature[ch] = feature;
					const float dL_dchannel_f = dL_dpixel_f[ch];
					dL_dalpha += (feature - accum_rec_f[ch]) * dL_dchannel_f;
					dL_dsampled_feature[ch] += dchannel_dcolor * dL_dchannel_f;
				}

				// Backprop through the tangent->world normal rotation (transpose).
				float dL_dsampled_feature_unit[3] = {0};
				dL_dsampled_feature_unit[0] = rotmat_row0.x * dL_dtexture_world_normal[0] + rotmat_row0.y * dL_dtexture_world_normal[1] + rotmat_row0.z * dL_dtexture_world_normal[2];
				dL_dsampled_feature_unit[1] = rotmat_row1.x * dL_dtexture_world_normal[0] + rotmat_row1.y * dL_dtexture_world_normal[1] + rotmat_row1.z * dL_dtexture_world_normal[2];
				dL_dsampled_feature_unit[2] = rotmat_row2.x * dL_dtexture_world_normal[0] + rotmat_row2.y * dL_dtexture_world_normal[1] + rotmat_row2.z * dL_dtexture_world_normal[2];

				// Backprop through the normalization of the sampled normal.
				dnormvdv(sampled_feature, dL_dsampled_feature_unit, dL_dsampled_feature);

				// Accumulate gradients w.r.t. the rotation matrix elements.
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 0], dL_dtexture_world_normal[0] * sampled_normal_unit[0]); // R00
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 1], dL_dtexture_world_normal[1] * sampled_normal_unit[0]); // R01
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 2], dL_dtexture_world_normal[2] * sampled_normal_unit[0]); // R02
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 3], dL_dtexture_world_normal[0] * sampled_normal_unit[1]); // R10
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 4], dL_dtexture_world_normal[1] * sampled_normal_unit[1]); // R11
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 5], dL_dtexture_world_normal[2] * sampled_normal_unit[1]); // R12
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 6], dL_dtexture_world_normal[0] * sampled_normal_unit[2]); // R20
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 7], dL_dtexture_world_normal[1] * sampled_normal_unit[2]); // R21
				atomicAdd(&dL_drotation_matrices[global_id * 9 + 8], dL_dtexture_world_normal[2] * sampled_normal_unit[2]); // R22

				// Backprop through bilinear texture sampling.
				const float2 dL_dtexCoords = bilinear_sample_backward(
					dL_dsampled_feature,
					texture_coords.x,
					texture_coords.y,
					texture_features + texture_offset,
					dL_dtexture_features + texture_offset,
					texture_size,
					TS);
				dL_ds_textures.x = dL_dtexCoords.x / 4.0f;
				dL_ds_textures.y = dL_dtexCoords.y / 4.0f;
			}
			else
			{
				// --- Baseline mode -----------------------------------------------
				for (int ch = 0; ch < S; ch++)
				{
					const float feature = collected_features[ch * BLOCK_SIZE + j];
					accum_rec_f[ch] = last_alpha * last_feature[ch] + (1.f - last_alpha) * accum_rec_f[ch];
					last_feature[ch] = feature;
					const float dL_dchannel_f = dL_dpixel_f[ch];
					dL_dalpha += (feature - accum_rec_f[ch]) * dL_dchannel_f;
					atomicAdd(&(dL_dsplat_features[global_id * S + ch]), dchannel_dcolor * dL_dchannel_f);
				}
			}

			float dL_dz = 0.0f;
			float dL_dweight = 0;
#if RENDER_AXUTILITY
			const float m_d = far_n / (far_n - near_n) * (1 - near_n / c_d);
			const float dmd_dd = (far_n * near_n) / ((far_n - near_n) * c_d * c_d);
			if (contributor == median_contributor - 1)
			{
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT
			// if not detached weight, sometimes
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			const float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals.
			// Only in baseline mode: with textures the rendered normal comes
			// from the texture normal map, not the per-Gaussian splat normal.
			if (TS == 0)
			{
				for (int ch = 0; ch < 3; ch++)
				{
					accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
					last_normal[ch] = normal[ch];
					dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
					atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
				}
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// In texture mode, disable alpha gradient propagation when it was
			// clamped at 0.99 (stabilises texture optimisation).
			if (TS > 0 && alpha == 0.99f)
			{
				dL_dalpha = 0.0f;
			}

			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
#if RENDER_AXUTILITY
			dL_dz += alpha * T * dL_ddepth;
#endif

			if (TS > 0 || rho3d <= rho2d)
			{
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T).
				// dL_ds_textures is zero in baseline mode, so this branch is
				// bit-for-bit identical to the baseline transMat path there.
				const float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x + dL_ds_textures.x,
					dL_dG * -G * s.y + dL_dz * Tw.y + dL_ds_textures.y};
				const float3 dz_dTw = {s.x, s.y, 1.0};
				const float dsx_pz = dL_ds.x / p.z;
				const float dsy_pz = dL_ds.y / p.z;
				const float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
				const float3 dL_dk = cross(l, dL_dp);
				const float3 dL_dl = cross(dL_dp, k);

				const float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				const float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				const float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x,
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y,
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};

				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dtransMat[global_id * 9 + 0], dL_dTu.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 1], dL_dTu.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 2], dL_dTu.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 3], dL_dTv.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 4], dL_dTv.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 5], dL_dTv.z);
				atomicAdd(&dL_dtransMat[global_id * 9 + 6], dL_dTw.x);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7], dL_dTw.y);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dTw.z);
			}
			else
			{
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				const float dG_ddelx = -G * FilterInvSquare * d.x;
				const float dG_ddely = -G * FilterInvSquare * d.y;
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				atomicAdd(&dL_dtransMat[global_id * 9 + 8], dL_dz);	   // propagate depth loss
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

__device__ void compute_transmat_aabb(
	int idx,
	const float *Ts_precomp,
	const float3 *p_origs,
	const glm::vec2 *scales,
	const glm::vec4 *rots,
	const float *projmatrix,
	const float *viewmatrix,
	const int W, const int H,
	const float3 *dL_dnormals,
	const float *dL_drotation_matrices,
	const float3 *dL_dmean2Ds,
	float *dL_dTs,
	glm::vec3 *dL_dmeans,
	glm::vec2 *dL_dscales,
	glm::vec4 *dL_drots)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;

	// Gradients accumulated onto the rotation matrix by texture normal
	// mapping. These are all zero in baseline mode (the buffer is zero-
	// initialised and never written by renderCUDA when TS == 0).
	glm::mat3 dL_drotTexmat = glm::mat3(
		dL_drotation_matrices[idx * 9 + 0], dL_drotation_matrices[idx * 9 + 1], dL_drotation_matrices[idx * 9 + 2],
		dL_drotation_matrices[idx * 9 + 3], dL_drotation_matrices[idx * 9 + 4], dL_drotation_matrices[idx * 9 + 5],
		dL_drotation_matrices[idx * 9 + 6], dL_drotation_matrices[idx * 9 + 7], dL_drotation_matrices[idx * 9 + 8]);

	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr)
	{
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]);
		normal = {0.0, 0.0, 0.0};
	}
	else
	{
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale = scales[idx];
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scale, 1.0f);

		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
			glm::vec4(L[0], 0.0),
			glm::vec4(L[1], 0.0),
			glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1));

		glm::mat4 world2ndc = glm::mat4(
			projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
			projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
			projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
			projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]);

		glm::mat3x4 ndc2pix = glm::mat3x4(
			glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W - 1) / 2.0),
			glm::vec4(0.0, float(H) / 2.0, 0.0, float(H - 1) / 2.0),
			glm::vec4(0.0, 0.0, 0.0, 1.0));

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx * 9 + 0], dL_dTs[idx * 9 + 1], dL_dTs[idx * 9 + 2],
		dL_dTs[idx * 9 + 3], dL_dTs[idx * 9 + 4], dL_dTs[idx * 9 + 5],
		dL_dTs[idx * 9 + 6], dL_dTs[idx * 9 + 7], dL_dTs[idx * 9 + 8]);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if (dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		const float distance = T[2].x * T[2].x + T[2].y * T[2].y - T[2].z * T[2].z;
		const float f = 1 / (distance);
		const float dpx_dT00 = f * T[2].x;
		const float dpx_dT01 = f * T[2].y;
		const float dpx_dT02 = -f * T[2].z;
		const float dpy_dT10 = f * T[2].x;
		const float dpy_dT11 = f * T[2].y;
		const float dpy_dT12 = -f * T[2].z;
		const float dpx_dT30 = T[0].x * (f - 2 * f * f * T[2].x * T[2].x);
		const float dpx_dT31 = T[0].y * (f - 2 * f * f * T[2].y * T[2].y);
		const float dpx_dT32 = -T[0].z * (f + 2 * f * f * T[2].z * T[2].z);
		const float dpy_dT30 = T[1].x * (f - 2 * f * f * T[2].x * T[2].x);
		const float dpy_dT31 = T[1].y * (f - 2 * f * f * T[2].y * T[2].y);
		const float dpy_dT32 = -T[1].z * (f + 2 * f * f * T[2].z * T[2].z);

		dL_dT[0].x += dL_dmean2D.x * dpx_dT00;
		dL_dT[0].y += dL_dmean2D.x * dpx_dT01;
		dL_dT[0].z += dL_dmean2D.x * dpx_dT02;
		dL_dT[1].x += dL_dmean2D.y * dpy_dT10;
		dL_dT[1].y += dL_dmean2D.y * dpy_dT11;
		dL_dT[1].z += dL_dmean2D.y * dpy_dT12;
		dL_dT[2].x += dL_dmean2D.x * dpx_dT30 + dL_dmean2D.y * dpy_dT30;
		dL_dT[2].y += dL_dmean2D.x * dpx_dT31 + dL_dmean2D.y * dpy_dT31;
		dL_dT[2].z += dL_dmean2D.x * dpx_dT32 + dL_dmean2D.y * dpy_dT32;

		if (Ts_precomp != nullptr)
		{
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}

	if (Ts_precomp != nullptr)
		return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1 : -1;
	dL_dtn = multiplier * dL_dtn;
	// Flip only the normal component (third row) of the rotation matrix
	// gradient to match the forward DUAL_VISIABLE flip.
	dL_drotTexmat[2] = multiplier * dL_drotTexmat[2];
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z));

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);

	// Add the gradients accumulated on the rotation matrix by texture normal
	// mapping (zero in baseline mode).
	dL_dR += dL_drotTexmat;

	dL_drots[idx] = quat_to_rotmat_vjp(rot, dL_dR);
	dL_dscales[idx] = glm::vec2(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1]));
	dL_dmeans[idx] = glm::vec3(dL_dM[2]);
}

template <int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3 *means3D,
	const float *transMats,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec2 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *viewmatrix,
	const float *projmatrix,
	const float focal_x,
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3 *campos,
	// grad input
	float *dL_dtransMats,
	const float *dL_dnormal3Ds,
	const float *dL_drotation_matrices,
	float *dL_dcolors,
	float *dL_dshs,
	float3 *dL_dmean2Ds,
	glm::vec3 *dL_dmean3Ds,
	glm::vec2 *dL_dscales,
	glm::vec4 *dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float *Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx,
		Ts_precomp,
		means3D, scales, rotations,
		projmatrix, viewmatrix, W, H,
		(float3 *)dL_dnormal3Ds,
		dL_drotation_matrices,
		dL_dmean2Ds,
		(dL_dtransMats),
		dL_dmean3Ds,
		dL_dscales,
		dL_drots);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3 *)means3D, *campos, shs, clamped, (glm::vec3 *)dL_dcolors, (glm::vec3 *)dL_dmean3Ds, (glm::vec3 *)dL_dshs);

	// hack the gradient here for densitification
	float depth = transMats[idx * 9 + 8];
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3 *means3D,
	const int *radii,
	const float *shs,
	const bool *clamped,
	const glm::vec2 *scales,
	const glm::vec4 *rotations,
	const float scale_modifier,
	const float *transMats,
	const float *viewmatrix,
	const float *projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3 *campos,
	float3 *dL_dmean2Ds,
	const float *dL_dnormal3Ds,
	float *dL_dtransMats,
	float *dL_drotation_matrices,
	float *dL_dcolors,
	float *dL_dshs,
	glm::vec3 *dL_dmean3Ds,
	glm::vec2 *dL_dscales,
	glm::vec4 *dL_drots)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		(float3 *)means3D,
		transMats,
		radii,
		shs,
		clamped,
		(glm::vec2 *)scales,
		(glm::vec4 *)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,
		dL_dtransMats,
		dL_dnormal3Ds,
		dL_drotation_matrices,
		dL_dcolors,
		dL_dshs,
		dL_dmean2Ds,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2 *ranges,
	const uint32_t *point_list,
	int S, int TS, int W, int H, // //
	float focal_x, float focal_y,
	const float *bg_color,
	const float2 *means2D,
	const float4 *normal_opacity,
	const float3 *rotation_matrices,
	const float *colors,
	const float *splat_features,   // //
	const float *texture_features, // //
	int texture_size,
	const float *transMats,
	const float *depths,
	const float *final_Ts,
	const uint32_t *n_contrib,
	const float *dL_dpixels,
	const float *dL_dpixels_features, // //
	const float *dL_depths,
	float *dL_dtransMat,
	float3 *dL_dmean2D,
	float *dL_dnormal3D,
	float *dL_dopacity,
	float *dL_dcolors,
	float *dL_dsplatfeature,	 // //
	float *dL_dtexturefeature,	 // //
	float *dL_drotation_matrices // //
)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		S, TS, W, H, // //
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_opacity,
		rotation_matrices,
		transMats,
		colors,
		splat_features,	  // //
		texture_features, // //
		texture_size,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dpixels_features, // //
		dL_depths,
		dL_dtransMat,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors,
		dL_dsplatfeature,	  // //
		dL_dtexturefeature,	  // //
		dL_drotation_matrices // //
	);
}
