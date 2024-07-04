#!/bin/bash

python_executable=python$1
pytorch_version=$2
cuda_version=$3

pip install --upgrade pip
# If we don't install before installing Pytorch, we get error for torch 2.0.1
# ERROR: Could not find a version that satisfies the requirement setuptools>=40.8.0 (from versions: none)
pip install lit
# For some reason torch 2.2.0 on python 3.12 errors saying no setuptools
pip install setuptools
# We want to figure out the CUDA version to download pytorch
# e.g. we can have system CUDA version being 11.7 but if torch==1.12 then we need to download the wheel from cu116
# see https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
# This code is ugly, maybe there's a better way to do this.
echo $MATRIX_CUDA_VERSION
echo $MATRIX_TORCH_VERSION
export TORCH_CUDA_VERSION=$(python -c "from os import environ as env; \
minv = {'1.12': 113, '1.13': 116, '2.0': 117, '2.1': 118, '2.2': 118, '2.3': 118, '2.4': 118}[env['MATRIX_TORCH_VERSION']]; \
maxv = {'1.12': 116, '1.13': 117, '2.0': 118, '2.1': 121, '2.2': 121, '2.3': 121, '2.4': 121}[env['MATRIX_TORCH_VERSION']]; \
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
