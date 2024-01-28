#pragma once

#include <stdint.h>
#include <torch/torch.h>


void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);