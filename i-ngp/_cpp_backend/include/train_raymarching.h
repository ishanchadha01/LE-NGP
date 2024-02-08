#pragma once

#include <stdint.h>
#include <torch/torch.h>

void train_raymarching(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float scale, const float dt_gamma, const uint32_t max_steps, const uint32_t num_rays, const uint32_t num_cascades, const uint32_t grid_size, const uint32_t max_points, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter);
