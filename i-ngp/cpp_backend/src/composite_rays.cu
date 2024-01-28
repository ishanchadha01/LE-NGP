#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}


template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const float T_thresh,
    int* rays_alive, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ sigmas, 
    const scalar_t* __restrict__ rgbs, 
    const scalar_t* __restrict__ deltas, 
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;
    const int index = rays_alive[n]; // ray id
    
    // locate 
    sigmas += n * n_step;
    rgbs += n * n_step * 3;
    deltas += n * n_step * 2;
    
    rays_t += index;
    weights_sum += index;
    depth += index;
    image += index * 3;
    scalar_t t = rays_t[0]; // current ray's t
    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t r = image[0];
    scalar_t g = image[1];
    scalar_t b = image[2];

    // accumulate 
    uint32_t step = 0;
    while (step < n_step) {
        
        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        /* 
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        --> 
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        step++;
    }

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t[0] = t;
    }
    
    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;
    image[0] = r;
    image[1] = g;
    image[2] = b;
}


void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    image.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, T_thresh, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}