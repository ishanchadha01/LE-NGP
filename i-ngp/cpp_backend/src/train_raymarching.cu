#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>
#include <limits>

#include "include/train_raymarching.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

inline constexpr __device__ float SQRT3() { return 1.7320508075688772f; }
inline constexpr __device__ float RSQRT3() { return 0.5773502691896258f; }
inline constexpr __device__ float PI() { return 3.141592653589793f; }
inline constexpr __device__ float RPI() { return 0.3183098861837907f; }


inline __host__ __device__ float clamp(const float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}


inline __host__ __device__ float signf(const float x) {
    return copysignf(1.0, x);
}


template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}


inline __device__ int mip_from_pos(const float x, const float y, const float z, const float max_cascade) {
    const float mx = fmaxf(fabsf(x), fmaxf(fabs(y), fabs(z)));
    int exponent;
    frexpf(mx, &exponent); // [0, 0.5) --> -1, [0.5, 1) --> 0, [1, 2) --> 1, [2, 4) --> 2, ...
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}


inline __device__ int mip_from_dt(const float dt, const float grid_size, const float max_cascade) {
    const float mx = dt * grid_size * 0.5;
    int exponent;
    frexpf(mx, &exponent);
    return fminf(max_cascade - 1, fmaxf(0, exponent));
}


// rays_o/d: [N, 3]
// grid: [CHHH / 8]
// xyzs, dirs, deltas: [M, 3], [M, 3], [M, 2]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const uint8_t * __restrict__ grid,
    const float scale,
    const float dt_gamma, const uint32_t max_steps,
    const uint32_t num_rays, const uint32_t num_cascades, 
    const uint32_t grid_size, const uint32_t max_points,
    const scalar_t* __restrict__ nears, 
    const scalar_t* __restrict__ fars,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * deltas,
    int * rays, 
    int * counter
) {
    // do 2 passes, first to estimate num of steps
    // take small steps if it seems occupied
    // second to march rays and write points

    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= num_rays) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    // ray marching
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float rH = 1 / (float)grid_size;
    const float H3 = grid_size * grid_size * grid_size;
    const float near = nears[n];
    const float far = fars[n];
    const float noise = noises[n];
    const float dt_min = 2 * SQRT3() / max_steps;
    const float dt_max = 2 * SQRT3() * (1 << (num_cascades - 1)) / grid_size;
    
    float t0 = near;
    
    // manually clamp TODO: could add noise here
    t0 += clamp(t0 * dt_gamma, dt_min, dt_max)

    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    while (t < far && num_steps < max_steps) {
        // current point
        const float x = clamp(ox + t * dx, -scale, scale);
        const float y = clamp(oy + t * dy, -scale, scale);
        const float z = clamp(oz + t * dz, -scale, scale);
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level, mipmaps are at different resolutions basically
        const int level = max(mip_from_pos(x, y, z, num_cascades), mip_from_dt(dt, grid_size, num_cascades)); // range in [0, num_cascades - 1]
        const float mip_bound = fminf(scalbnf(1.0f, level), scale);
        const float mip_rbound = 1 / mip_bound;
        
        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));
        const uint32_t index = level * H3 + __morton3D(nx, ny, nz);
        const bool occ = grid[index / 8] & (1 << (index % 8));

        // if occpuied, advance a small step
        if (occ) {
            num_steps++;
            t += dt;
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

    // second pass: really locate and write points & dirs
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);

    // write rays
    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = num_steps;

    if (num_steps == 0) return;
    if (point_index + num_steps > M) return;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index * 2;
    t = t0;
    uint32_t step = 0;
    float last_t = t;

    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -scale, scale);
        const float y = clamp(oy + t * dy, -scale, scale);
        const float z = clamp(oz + t * dz, -scale, scale);
        const float dt = clamp(t * dt_gamma, dt_min, dt_max);

        // get mip level
        const int level = max(mip_from_pos(x, y, z, num_cascades), mip_from_dt(dt, grid_size, num_cascades)); // range in [0, num_cascades - 1]
        const float mip_bound = fminf(scalbnf(1.0f, level), scale);
        const float mip_rbound = 1 / mip_bound;

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));
        const int ny = clamp(0.5 * (y * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));
        const int nz = clamp(0.5 * (z * mip_rbound + 1) * H, 0.0f, (float)(grid_size - 1));

        // query grid
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
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
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

void train_raymarching(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float scale, const float dt_gamma, const uint32_t max_steps, const uint32_t num_rays, const uint32_t num_cascades, const uint32_t grid_size, const uint32_t max_points, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<uint8_t>(), scale, dt_gamma, max_steps, num_rays, num_cascades, grid_size, max_points, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>());
    }));
}
