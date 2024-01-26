#pragma once

#include <stdint.h>
#include <torch/torch.h>


void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void invert_morton3D(const at::Tensor indices, const uint32_t N, at::Tensor coords);
void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield);
void ray_intersection(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t num_rays, const float min_near, at::Tensor nears, at::Tensor fars);
