#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

distance_module = Extension('distances',
                            sources=['./distances.cpp'],
                            include_dirs=[numpy.get_include()],
                            language="c++",
                            extra_compile_args=["-O2", "-march=native", "-fopenmp", "-mavx", "-msse", "-mfma",
                                                "-funroll-loops"],
                            libraries=["gomp"])

setup(
    name='distances',
    version='1.0',
    description='here is the description',
    ext_modules=[distance_module]
)
