#pragma once

#include <stdint.h>
#include <torch/torch.h>


void train_composite_rays_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const uint32_t max_points, const uint32_t num_rays, const float T_thresh, at::Tensor weights_sum, at::Tensor depth, at::Tensor image);
void train_composite_rays_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor image, const uint32_t max_points, const uint32_t num_rays, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs);
