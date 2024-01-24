#pragma once

#include <stdint.h>
#include <torch/torch.h>


void morton3D(const at::Tensor coords, const uint32_t N, at::Tensor indices);
void morton3D_invert(const at::Tensor indices, const uint32_t N, at::Tensor coords);