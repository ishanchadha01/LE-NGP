#pragma once

#include <stdint.h>
#include <torch/torch.h>


void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void invert_morton3D(const at::Tensor indices, const uint32_t N, at::Tensor coords);
void packbits(const at::Tensor grid, const uint32_t N, const float density_thresh, at::Tensor bitfield);
