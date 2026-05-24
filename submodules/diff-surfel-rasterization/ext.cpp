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

#include <torch/extension.h>
#include "rasterize_points.h"
#include "atlas.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Per-primitive texture atlas for hardware bilinear sampling (eval-time
  // only). Construct with a [P, C, T, T] CUDA float32 tensor, pass the
  // instance to `GaussianRasterizer.forward(..., atlas=)`. The rasterizer
  // pulls the cudaTextureObjects out via TextureAtlas::view() internally.
  py::class_<TextureAtlas>(m, "TextureAtlas")
        .def(py::init<const torch::Tensor&>(), py::arg("texture_features"),
             "Build the atlas from a [P, C, T, T] CUDA float32 tensor.")
        .def("parameters", &TextureAtlas::parameters,
             "Return layout parameters (P, C, T, charts_per_dim, channels1, channels2).");

  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}
