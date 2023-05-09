from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

cpu_module = CppExtension(name='distance', 
                            sources=['distance.cpp'], 
                            extra_compile_args=['-std=c++14', '-fopenmp', '-mavx', '-msse', '-ftree-vectorize'], 
                            libraries=['gomp'])

setup(
    name='distance', 
    ext_modules=[cpu_module], 
    cmdclass={
        'build_ext': BuildExtension
    }
)
