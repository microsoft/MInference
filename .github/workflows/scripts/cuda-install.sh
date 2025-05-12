#!/bin/bash

# Replace '.' with '-' ex: 11.8 -> 11-8
cuda_version=$(echo $1 | tr "." "-")
# Removes '-' and '.' ex: ubuntu-20.04 -> ubuntu2004
OS=$(echo $2 | tr -d ".\-")

ARCH=$(uname -m)
ARCH_TYPE=$ARCH

# Detectar si es Tegra
if [[ "$ARCH" == "aarch64" ]]; then
    if uname -a | grep -qi tegra; then
        ARCH_TYPE="tegra-aarch64"
    fi
fi

echo "Detected architecture: ${ARCH_TYPE}"

# Installs CUDA
if [[ "$ARCH_TYPE" == "tegra-aarch64" ]]; then
    # Jetson (Tegra)
    wget -nv \
        https://developer.download.nvidia.com/compute/cuda/repos/${OS}/arm64/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600

elif [[ "$ARCH_TYPE" == "tegra-aarch64" ]]; then
    # Jetson (Tegra)
    wget -nv \
        https://developer.download.nvidia.com/compute/cuda/repos/${OS}/arm64/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600
else
    # ARM64 SBSA (Grace)
    wget -nv \
        https://developer.download.nvidia.com/compute/cuda/repos/${OS}/sbsa/cuda-${DISTRO}.pin \
        -O /etc/apt/preferences.d/cuda-repository-pin-600
fi

sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb
sudo apt -qq update
sudo apt -y install cuda-${cuda_version} cuda-nvcc-${cuda_version} cuda-libraries-dev-${cuda_version}
sudo apt clean

# Test nvcc
PATH=/usr/local/cuda-$1/bin:${PATH}
nvcc --version

# Log gcc, g++, c++ versions
gcc --version
g++ --version
c++ --version
