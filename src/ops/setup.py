from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Check if NVCC is available, otherwise skip build or provide dummy
if os.system("nvcc --version") == 0:
    ext_modules = [
        CUDAExtension(
            name='src.ops.cuda_ops_ext',
            sources=['src/ops/bindings.cpp', 'src/ops/projection.cu'],
        )
    ]
else:
    print("WARNING: NVCC not found. Skipping CUDA extension build.")
    ext_modules = []

setup(
    name='cuda_ops',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
