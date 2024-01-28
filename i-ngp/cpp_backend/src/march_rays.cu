#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#include "include/march_rays.h"

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

inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}


template <typename scalar_t>
__global__ void kernel_march_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const int* __restrict__ rays_alive, 
    const scalar_t* __restrict__ rays_t, 
    const scalar_t* __restrict__ rays_o, 
    const scalar_t* __restrict__ rays_d, 
    const float scale,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t num_cascades, const uint32_t grid_size,
    const uint8_t * __restrict__ grid,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    scalar_t* xyzs, scalar_t* dirs, scalar_t* deltas
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;
    const int index = rays_alive[n]; // ray id
    const float noise = noises[n];
    
    // locate
    rays_o += index * 3;
    rays_d += index * 3;
    xyzs += n * n_step * 3;
    dirs += n * n_step * 3;
    deltas += n * n_step * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)grid_size;
    const float H3 = grid_size * grid_size * grid_size;
    float t = rays_t[index]; // current ray's t
    const float near = nears[index], far = fars[index];
    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (num_cascades - 1)) / grid_size;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness
    t += clamp(t * dt_gamma, dt_min, dt_max) * noise;
    float last_t = t;

    while (t < far && step < n_step) {
        // current point
        const float x = clamp(ox + t * dx, -scale, scale);
        const float y = clamp(oy + t * dy, -scale, scale);
        const float z = clamp(oz + t * dz, -scale, scale);
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, num_cascades), mip_from_dt(dt, grid_size, num_cascades)); // range in [0, C - 1]
        const float mip_bound = fminf(scalbnf(1, level), scale);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step, and write to output
        if (occ) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            // calc dt
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            // step
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) * rH * 2 - 1) * mip_bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) * rH * 2 - 1) * mip_bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) * rH * 2 - 1) * mip_bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                t += clamp(t * dt_gamma, dt_min, dt_max);
            } while (t < tt);
        }
    }
}

void march_rays(const uint32_t n_alive, const uint32_t n_step, const at::Tensor rays_alive, const at::Tensor rays_t, const at::Tensor rays_o, const at::Tensor rays_d, const float scale, const float dt_gamma, const uint32_t max_steps, const uint32_t num_cascades, const uint32_t grid_size, const at::Tensor grid, const at::Tensor near, const at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays", ([&] {
        kernel_march_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), scale, dt_gamma, max_steps, num_cascades, grid_size, grid.data_ptr<uint8_t>(), near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>());
    }));
}