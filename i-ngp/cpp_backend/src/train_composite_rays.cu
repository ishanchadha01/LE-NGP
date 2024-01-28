#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#include "include/train_composite_rays.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N], final pixel alpha
// depth: [N,]
// image: [N, 3]
template <typename scalar_t>
__global__ void kernel_train_composite_rays_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const uint32_t max_points, const uint32_t num_rays, 
    const float T_thresh, 
    scalar_t * weights_sum,
    scalar_t * depth,
    scalar_t * image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= num_rays) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps > max_points) {
        weights_sum[index] = 0;
        depth[index] = 0;
        image[index * 3] = 0;
        image[index * 3 + 1] = 0;
        image[index * 3 + 2] = 0;
        return;
    }
    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset * 2;

    // accumulate 
    uint32_t step = 0;
    scalar_t T = 1.0f;
    scalar_t r = 0, g = 0, b = 0, ws = 0, t = 0, d = 0;
    while (step < num_steps) {
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];
        t += deltas[1]; // real delta
        d += weight * t;
        ws += weight;
        T *= 1.0f - alpha;

        // minimal remained transmittence
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;

        step++;
    }
    // write
    weights_sum[index] = ws; // weights_sum
    depth[index] = d;
    image[index * 3] = r;
    image[index * 3 + 1] = g;
    image[index * 3 + 2] = b;
}


void train_composite_rays_forward(const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const uint32_t max_points, const uint32_t num_rays, const float T_thresh, at::Tensor weights_sum, at::Tensor depth, at::Tensor image) {

    static constexpr uint32_t N_THREAD = 128;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_forward", ([&] {
        kernel_composite_rays_train_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), max_points, num_rays, T_thresh, weights_sum.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}


// grad_weights_sum: [N,]
// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M, 2]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here 
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_train_composite_rays_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad_image,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs, 
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ image,
    const uint32_t max_points, const uint32_t num_rays, const float T_thresh,
    scalar_t * grad_sigmas,
    scalar_t * grad_rgbs
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= num_rays) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps > max_points) return;

    grad_weights_sum += index;
    grad_image += index * 3;
    weights_sum += index;
    image += index * 3;
    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset * 2;
    grad_sigmas += offset;
    grad_rgbs += offset * 3;

    // accumulate 
    uint32_t step = 0;
    
    scalar_t T = 1.0f;
    const scalar_t r_final = image[0], g_final = image[1], b_final = image[2], ws_final = weights_sum[0];
    scalar_t r = 0, g = 0, b = 0, ws = 0;

    while (step < num_steps) {
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];
        ws += weight;
        T *= 1.0f - alpha;
        
        // check https://note.kiui.moe/others/nerf_gradient/ for the gradient calculation.
        // write grad_rgbs
        grad_rgbs[0] = grad_image[0] * weight;
        grad_rgbs[1] = grad_image[1] * weight;
        grad_rgbs[2] = grad_image[2] * weight;

        // write grad_sigmas
        grad_sigmas[0] = deltas[0] * (
            grad_image[0] * (T * rgbs[0] - (r_final - r)) + 
            grad_image[1] * (T * rgbs[1] - (g_final - g)) + 
            grad_image[2] * (T * rgbs[2] - (b_final - b)) +
            grad_weights_sum[0] * (1 - ws_final)
        );

        // minimal remained transmittence
        if (T < T_thresh) break;
        
        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        grad_sigmas++;
        grad_rgbs += 3;
        step++;
    }
}

void train_composite_rays_backward(const at::Tensor grad_weights_sum, const at::Tensor grad_image, const at::Tensor sigmas, const at::Tensor rgbs, const at::Tensor deltas, const at::Tensor rays, const at::Tensor weights_sum, const at::Tensor image, const uint32_t max_points, const uint32_t num_rays, const float T_thresh, at::Tensor grad_sigmas, at::Tensor grad_rgbs) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_image.scalar_type(), "composite_rays_train_backward", ([&] {
        kernel_composite_rays_train_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad_weights_sum.data_ptr<scalar_t>(), grad_image.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), weights_sum.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(), max_points, num_rays, T_thresh, grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.data_ptr<scalar_t>());
    }));
}