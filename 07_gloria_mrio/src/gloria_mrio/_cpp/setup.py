"""
Build script for C++ compute kernel.

Usage:
    cd src/gloria_mrio/_cpp
    python setup.py build_ext --inplace
    cp _compute_kernel*.so ../     # Linux
    cp _compute_kernel*.pyd ../    # Windows
    
Or directly with pip:
    pip install ./src/gloria_mrio/_cpp/

Prerequisites:
    pip install pybind11
    
For CBLAS support (optional, ~2x faster matmul):
    # On Ubuntu/Debian:
    apt-get install libopenblas-dev
    # Then build with:
    CFLAGS="-DUSE_CBLAS" LDFLAGS="-lopenblas" python setup.py build_ext --inplace
    
For Docker container (python:3.11-slim):
    apt-get update && apt-get install -y build-essential libopenblas-dev
    pip install pybind11
    cd src/gloria_mrio/_cpp && python setup.py build_ext --inplace && cp _compute_kernel*.so ../
"""

import os
import sys
from setuptools import setup, Extension
import pybind11

extra_compile_args = ['-O3', '-ffast-math', '-march=native']
extra_link_args = []

# OpenMP support
if sys.platform == 'darwin':
    # macOS with Homebrew libomp
    extra_compile_args += ['-Xpreprocessor', '-fopenmp']
    extra_link_args += ['-lomp']
else:
    # Linux
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

# Optional CBLAS support
if os.environ.get('USE_CBLAS') or os.environ.get('CFLAGS', '').find('USE_CBLAS') >= 0:
    extra_compile_args.append('-DUSE_CBLAS')
    extra_link_args.append('-lopenblas')

ext = Extension(
    '_compute_kernel',
    sources=['compute_kernel.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=extra_compile_args + ['-std=c++17'],
    extra_link_args=extra_link_args,
)

setup(
    name='gloria_compute_kernel',
    version='1.0',
    ext_modules=[ext],
)
