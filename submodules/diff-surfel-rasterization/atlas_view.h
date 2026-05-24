#pragma once

#include "cuda_runtime.h"

// POD view of a per-primitive texture atlas, passed by value to the rasterizer
// kernel. Default-constructs to a disabled state (both texture objects 0); the
// kernel checks `enabled()` to decide between hardware and software sampling.
//
// Chart coordinates for primitive p are recovered as (p % charts_per_dim,
// p / charts_per_dim) — no indirection buffer.
struct AtlasView {
    cudaTextureObject_t tex_ch1 = 0;  // channels [0, channels1)
    cudaTextureObject_t tex_ch2 = 0;  // channels [channels1, channels1 + channels2)
    int T = 0;                         // per-primitive chart edge length in texels
    int charts_per_dim = 0;            // sqrt(ceil) of P
    int channels1 = 0;                 // <= 4
    int channels2 = 0;                 // <= 4

    __host__ __device__ bool enabled() const { return tex_ch1 != 0 || tex_ch2 != 0; }
};
