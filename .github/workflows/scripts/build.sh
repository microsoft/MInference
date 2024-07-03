#!/bin/bash

python_executable=python$1
cuda_home=/usr/local/cuda-$2

# Update paths
PATH=${cuda_home}/bin:$PATH
LD_LIBRARY_PATH=${cuda_home}/lib64:$LD_LIBRARY_PATH

# Install requirements
$python_executable -m pip install wheel packaging
$python_executable -m pip install flash_attn triton

# Limit the number of parallel jobs to avoid OOM
export MAX_JOBS=1
# Make sure release wheels are built for the following architectures
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
# Build
if [ "$3" = sdist ];
then
MINFERENCE_SKIP_CUDA_BUILD="TRUE" $python_executable setup.py $3 --dist-dir=dist
else
MINFERENCE_LOCAL_VERSION=cu${MATRIX_CUDA_VERSION}torch${MATRIX_TORCH_VERSION} MINFERENCE_FORCE_BUILD="TRUE" $python_executable setup.py $3 --dist-dir=dist
fi