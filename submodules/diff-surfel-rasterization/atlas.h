#pragma once

#include <torch/extension.h>
#include <map>
#include <string>

#include "atlas_view.h"

// Builds and owns the cudaArrays + cudaTextureObjects that back hardware
// bilinear sampling of per-primitive texture maps. Exposed to Python as
// `TextureAtlas`; construct with a [P, C, T, T] CUDA float32 tensor and pass
// the instance to `GaussianRasterizer.forward(..., atlas=)`.
//
// The atlas owns its CUDA resources for the object's lifetime — there is no
// reinitialise / rebuild path. The Python side caches one instance per model.
class TextureAtlas {
public:
    explicit TextureAtlas(const torch::Tensor& texture_features);
    ~TextureAtlas();

    TextureAtlas(const TextureAtlas&) = delete;
    TextureAtlas& operator=(const TextureAtlas&) = delete;

    AtlasView view() const {
        return AtlasView{tex_ch1_, tex_ch2_, T_, charts_per_dim_, channels1_, channels2_};
    }

    // Diagnostics only — surfaced through pybind11 for unit tests / printouts.
    std::map<std::string, int> parameters() const;

private:
    cudaArray_t array_ch1_ = nullptr;
    cudaArray_t array_ch2_ = nullptr;
    cudaTextureObject_t tex_ch1_ = 0;
    cudaTextureObject_t tex_ch2_ = 0;
    int P_ = 0;
    int C_ = 0;
    int T_ = 0;
    int charts_per_dim_ = 0;
    int channels1_ = 0;
    int channels2_ = 0;
};
