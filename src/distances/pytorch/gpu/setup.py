from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

gpu_module = CUDAExtension(name='distance', 
                            sources=['distance.cpp', 'kernels.cu'])

setup(
    name='distance', 
    ext_modules=[gpu_module], 
    cmdclass={
        'build_ext': BuildExtension
    }
)