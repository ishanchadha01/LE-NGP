#include <torch.extension.h>

#include "include/morton3D.h"
#include "include/ray_intersection.h"
#include "include/train_raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("invert_morton3D", &morton3D_invert, "invert_morton3D (CUDA)");
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("ray_intersection", &near_far_from_aabb, "ray_intersection (CUDA)");
    m.def("train_raymarching", &train_raymarching, "train_raymarching (CUDA)")
}