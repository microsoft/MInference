#!/bin/bash

# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

python_executable=python$1
pytorch_version=$2
cuda_version=$3

pip install --upgrade pip
# For some reason torch 2.2.0 on python 3.12 errors saying no setuptools
pip install setuptools==75.8.0
# With python 3.13 and torch 2.5.1, unless we update typing-extensions, we get error
# AttributeError: attribute '__default__' of 'typing.ParamSpec' objects is not writable
pip install typing-extensions==4.12.2
# We want to figure out the CUDA version to download pytorch
# e.g. we can have system CUDA version being 11.7 but if torch==1.12 then we need to download the wheel from cu116
# see https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
# This code is ugly, maybe there's a better way to do this.
echo $MATRIX_CUDA_VERSION
echo $MATRIX_TORCH_VERSION
export TORCH_CUDA_VERSION=$(python -c "from os import environ as env; \
minv = {'2.2': 118, '2.3': 118, '2.4': 118, '2.5': 118, '2.6': 118}[env['MATRIX_TORCH_VERSION']]; \
maxv = {'2.2': 121, '2.3': 121, '2.4': 124, '2.5': 124, '2.6': 124}[env['MATRIX_TORCH_VERSION']]; \
print(max(min(int(env['MATRIX_CUDA_VERSION']), maxv), minv))" \
)
if [[ ${pytorch_version} == *"dev"* ]]; then
pip install --no-cache-dir --pre torch==${pytorch_version} --index-url https://download.pytorch.org/whl/nightly/cu${TORCH_CUDA_VERSION}
else
pip install --no-cache-dir torch==${pytorch_version} --index-url https://download.pytorch.org/whl/cu${TORCH_CUDA_VERSION}
fi
nvcc --version
$python_executable --version
$python_executable -c "import torch; print('PyTorch:', torch.__version__)"
$python_executable -c "import torch; print('CUDA:', torch.version.cuda)"
$python_executable -c "from torch.utils import cpp_extension; print (cpp_extension.CUDA_HOME)"
