#!/usr/bin/bash
set -e  # Exit on first error

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
echo $BASE_DIR
PIP="$(which pip)"

if command -v nvidia-smi
then
    $PIP install ninja cmake wheel pybind11
    $PIP install -r "${BASE_DIR}/requirements.txt"
    $PIP install git+https://github.com/microsoft/nnscaler.git@2368540417bc3b77b7e714d3f1a0de8a51bb66e8
    $PIP install "rotary-emb @ git+https://github.com/Dao-AILab/flash-attention.git@9356a1c0389660d7e231ff3163c1ac17d9e3824a#subdirectory=csrc/rotary"
    $PIP install "block_sparse_attn @ git+https://github.com/HalberdOfPineapple/flash-attention.git@block-sparse"
    $PIP install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4.post1
    $PIP install torch==2.6.0 torchvision==0.21.0 
    $PIP install triton==3.0.0
elif command -v rocm-smi # TODO: to verify the correctness of dependencies in ROCm environment
then
    $PIP install ninja cmake wheel pybind11
    $PIP install --pre torch==2.3.1+rocm6.0 --index-url https://download.pytorch.org/whl/rocm6.0
    $PIP install git+https://github.com/OpenAI/triton.git@e192dba#subdirectory=python
    $PIP install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4.post1
    $PIP install -r "${BASE_DIR}/requirements.txt"
    $PIP install git+https://github.com/microsoft/nnscaler.git@2368540417bc3b77b7e714d3f1a0de8a51bb66e8
else
    echo "ERROR: both nvidia-smi and rocm-smi not found"
    exit 1
fi

# Get the path to nnscaler and write its path to PYTHONPATH in ~/.profile
NNSCALER_HOME=$(python -c "import nnscaler; print(nnscaler.__path__[0])")
echo "export NNSCALER_HOME=${NNSCALER_HOME}" >> ~/.profile
echo "export PYTHONPATH=${NNSCALER_HOME}:\${PYTHONPATH}" >> ~/.profile
source ~/.profile
$PIP install -e $BASE_DIR