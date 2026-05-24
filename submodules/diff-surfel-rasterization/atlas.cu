#include "atlas.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("TextureAtlas CUDA error: ") + \
                                 cudaGetErrorString(err) + " (" __FILE__ ":" + \
                                 std::to_string(__LINE__) + ")"); \
    } \
} while (0)

namespace {

// `charts_per_dim = ceil(sqrt(P))` packs P charts of size TxT into a square
// atlas. We bound it against the device's `maxTexture2D[0]` (clamped to 16384,
// which covers every Ampere/Hopper GPU we run on).
int compute_charts_per_dim(int P, int T) {
    cudaDeviceProp props;
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
    const int practical_max = std::min(props.maxTexture2D[0], 16384);
    if (T > practical_max) {
        throw std::runtime_error("TextureAtlas: T exceeds device max texture dim");
    }
    const int max_charts = practical_max / T;
    const int cpd = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(P))));
    if (cpd > max_charts) {
        throw std::runtime_error("TextureAtlas: P does not fit in a single atlas at this T");
    }
    return cpd;
}

// Pack one channel group directly into its cudaArray via surface writes. One
// block per primitive, T*T threads per block; channel coords are derived from
// the block index (chart_x = p % cpd, chart_y = p / cpd) so no indirection
// buffer is needed. T_Data is float / float2 / float4 to match the cudaArray
// format; channel_count <= sizeof(T_Data)/sizeof(float). Channels beyond
// channel_count are zero-padded.
template <typename T_Data>
__global__ void pack_atlas_group_kernel(
    const float* __restrict__ input_textures,
    cudaSurfaceObject_t surf,
    int P, int C_total, int T, int charts_per_dim,
    int channel_start, int channel_count)
{
    const int p = blockIdx.x;
    if (p >= P) return;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    if (tx >= T || ty >= T) return;

    const int chart_x = p % charts_per_dim;
    const int chart_y = p / charts_per_dim;
    const int texels_per_texture = T * T;
    const int texel_idx = ty * T + tx;

    const float* prim_group = input_textures
        + (size_t)p * C_total * texels_per_texture
        + (size_t)channel_start * texels_per_texture;

    T_Data out{};
    float* out_comps = reinterpret_cast<float*>(&out);
    constexpr int max_c = sizeof(T_Data) / sizeof(float);
    #pragma unroll
    for (int c = 0; c < max_c; ++c) {
        out_comps[c] = (c < channel_count)
            ? prim_group[c * texels_per_texture + texel_idx]
            : 0.0f;
    }

    const int atlas_x = chart_x * T + tx;
    const int atlas_y = chart_y * T + ty;
    surf2Dwrite(out, surf, atlas_x * sizeof(T_Data), atlas_y);
}

cudaArray_t allocate_array(int channels, int dim) {
    cudaChannelFormatDesc desc;
    switch (channels) {
        case 1: desc = cudaCreateChannelDesc<float>();  break;
        case 2: desc = cudaCreateChannelDesc<float2>(); break;
        // 3 or 4: cudaArrays of 3 floats aren't supported; use float4 and pad.
        default: desc = cudaCreateChannelDesc<float4>(); break;
    }
    cudaArray_t array = nullptr;
    CUDA_CHECK(cudaMallocArray(&array, &desc, dim, dim));
    return array;
}

cudaSurfaceObject_t make_surface(cudaArray_t array) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;
    cudaSurfaceObject_t surf = 0;
    CUDA_CHECK(cudaCreateSurfaceObject(&surf, &res));
    return surf;
}

cudaTextureObject_t make_texture(cudaArray_t array) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;
    cudaTextureDesc tex{};
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.readMode = cudaReadModeElementType;
    tex.normalizedCoords = 1;
    cudaTextureObject_t obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&obj, &res, &tex, nullptr));
    return obj;
}

// Launches the templated packer for the correct surface element type.
void launch_pack(const float* d_input, cudaSurfaceObject_t surf,
                 int P, int C_total, int T, int charts_per_dim,
                 int channel_start, int channel_count)
{
    if (channel_count <= 0) return;
    const int threads = T * T;
    if (threads > 1024) {
        throw std::runtime_error(
            "TextureAtlas: T*T > 1024 (T=" + std::to_string(T) + ") not supported");
    }
    dim3 grid(P, 1, 1);
    dim3 block(T, T, 1);

    if (channel_count == 1) {
        pack_atlas_group_kernel<float ><<<grid, block>>>(
            d_input, surf, P, C_total, T, charts_per_dim, channel_start, channel_count);
    } else if (channel_count == 2) {
        pack_atlas_group_kernel<float2><<<grid, block>>>(
            d_input, surf, P, C_total, T, charts_per_dim, channel_start, channel_count);
    } else {
        // 3 or 4 → float4 (the 3-channel case writes a zero in component 3).
        pack_atlas_group_kernel<float4><<<grid, block>>>(
            d_input, surf, P, C_total, T, charts_per_dim, channel_start, channel_count);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

TextureAtlas::TextureAtlas(const torch::Tensor& texture_features) {
    TORCH_CHECK(texture_features.is_cuda(),
                "TextureAtlas: texture_features must be a CUDA tensor");
    TORCH_CHECK(texture_features.dim() == 4,
                "TextureAtlas: texture_features must have shape [P, C, T, T]");
    TORCH_CHECK(texture_features.scalar_type() == torch::kFloat32,
                "TextureAtlas: texture_features must be float32");

    P_ = static_cast<int>(texture_features.size(0));
    C_ = static_cast<int>(texture_features.size(1));
    T_ = static_cast<int>(texture_features.size(2));
    TORCH_CHECK(static_cast<int>(texture_features.size(3)) == T_,
                "TextureAtlas: per-primitive chart must be square (T x T)");
    TORCH_CHECK(P_ > 0 && C_ > 0 && T_ > 0,
                "TextureAtlas: P, C, T must all be positive");
    TORCH_CHECK(C_ <= 8,
                "TextureAtlas: C must be in [1, 8] (got " + std::to_string(C_) + ")");

    channels1_ = std::min(C_, 4);
    channels2_ = std::max(0, C_ - 4);
    charts_per_dim_ = compute_charts_per_dim(P_, T_);
    const int atlas_dim = charts_per_dim_ * T_;

    const torch::Tensor t = texture_features.contiguous();
    const float* d_input = t.data_ptr<float>();

    array_ch1_ = allocate_array(channels1_, atlas_dim);
    cudaSurfaceObject_t surf1 = make_surface(array_ch1_);
    cudaSurfaceObject_t surf2 = 0;
    if (channels2_ > 0) {
        array_ch2_ = allocate_array(channels2_, atlas_dim);
        surf2 = make_surface(array_ch2_);
    }

    try {
        launch_pack(d_input, surf1, P_, C_, T_, charts_per_dim_,
                    /*channel_start=*/0, channels1_);
        if (channels2_ > 0) {
            launch_pack(d_input, surf2, P_, C_, T_, charts_per_dim_,
                        /*channel_start=*/channels1_, channels2_);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        if (surf1) cudaDestroySurfaceObject(surf1);
        if (surf2) cudaDestroySurfaceObject(surf2);
        if (array_ch1_) { cudaFreeArray(array_ch1_); array_ch1_ = nullptr; }
        if (array_ch2_) { cudaFreeArray(array_ch2_); array_ch2_ = nullptr; }
        throw;
    }

    cudaDestroySurfaceObject(surf1);
    if (surf2) cudaDestroySurfaceObject(surf2);

    tex_ch1_ = make_texture(array_ch1_);
    if (array_ch2_) tex_ch2_ = make_texture(array_ch2_);
}

TextureAtlas::~TextureAtlas() {
    if (tex_ch1_) cudaDestroyTextureObject(tex_ch1_);
    if (tex_ch2_) cudaDestroyTextureObject(tex_ch2_);
    if (array_ch1_) cudaFreeArray(array_ch1_);
    if (array_ch2_) cudaFreeArray(array_ch2_);
}

std::map<std::string, int> TextureAtlas::parameters() const {
    return {
        {"P", P_},
        {"C", C_},
        {"T", T_},
        {"charts_per_dim", charts_per_dim_},
        {"channels1", channels1_},
        {"channels2", channels2_},
    };
}
