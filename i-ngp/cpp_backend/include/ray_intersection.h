#pragma once

#include <stdint.h>
#include <torch/torch.h>

void ray_intersection(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t num_rays, const float min_near, at::Tensor nears, at::Tensor fars);
