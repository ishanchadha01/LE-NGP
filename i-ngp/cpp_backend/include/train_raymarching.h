#pragma once

#include <stdint.h>
#include <torch/torch.h>

void train_raymarching(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float scale, const float dt_gamma, const uint32_t max_steps, const uint32_t num_rays, const uint32_t num_cascades, const uint32_t grid_size, const uint32_t max_points, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter);
// void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t C, const uint32_t H, const at::Tensor grid, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor noises);