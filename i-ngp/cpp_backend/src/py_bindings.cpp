#include <torch.extension.h>

#include "include/morton3D.h"
#include "include/ray_intersection.h"
#include "include/train_raymarching.h"
#include "include/train_composite_rays.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("invert_morton3D", &morton3D_invert, "invert_morton3D (CUDA)");
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("ray_intersection", &near_far_from_aabb, "ray_intersection (CUDA)");
    m.def("train_raymarching", &train_raymarching, "train_raymarching (CUDA)")
    m.def("train_composite_rays_forward", &train_composite_rays_forward, "train_composite_rays_forward (CUDA)")
    m.def("train_composite_rays_backward", &train_composite_rays_backward, "train_composite_rays_backward (CUDA)")
    m.def("march_rays", &march_rays, "march_rays (CUDA)")
}