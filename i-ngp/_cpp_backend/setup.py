import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


'''
Usage:

python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)

python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)

python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)

'''
_src_path = os.path.dirname(os.path.abspath(__file__))
nvcc_flags = [
    '-O3', '-std=c++17',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]
c_flags = ['-O3', '-std=c++17'] # different on windows
include_dirs = [os.path.join(os.path.dirname(_src_path), '_cpp_backend/include')]
setup(
    name='cpp_backend', # package name, import this to use python API
    ext_modules=[
        CUDAExtension(
            name='cpp_backend', # extension name, import this to use CUDA API
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'morton3D.cu',
                'ray_intersection.cu',
                'composite_rays.cu',
                'march_rays.cu',
                'train_composite_rays.cu',
                'train_raymarching.cu',
                'py_bindings.cpp',
            ]],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            },
            include_dirs=include_dirs
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)