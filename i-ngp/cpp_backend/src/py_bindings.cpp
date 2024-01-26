#include <torch.extension.h>

#include "include/morton3D.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("invert_morton3D", &morton3D_invert, "invert_morton3D (CUDA)");
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("ray_intersection", &near_far_from_aabb, "ray_intersection (CUDA)");
}