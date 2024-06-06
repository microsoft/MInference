# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = {}  # type: ignore
with open("minference/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

INSTALL_REQUIRES = [
    "transformers>=4.37.0",
    "accelerate",
    "torch",
    "triton",
    "flash_attn",
    "pycuda==2023.1",
]
QUANLITY_REQUIRES = [
    "black==21.4b0",
    "flake8>=3.8.3",
    "isort>=5.5.4",
    "pre-commit",
    "pytest",
    "pytest-xdist",
]
DEV_REQUIRES = INSTALL_REQUIRES + QUANLITY_REQUIRES

ext_modules = [
    CUDAExtension(
        name="minference.cuda",
        sources=[
            os.path.join("csrc", "kernels.cpp"),
            os.path.join("csrc", "vertical_slash_index.cu"),
        ],
        extra_compile_args=["-std=c++17", "-O3"],
    )
]

setup(
    name="minference",
    version=VERSION["VERSION"],
    author="The MInference team",
    author_email="hjiang@microsoft.com",
    description="To speed up Long-context LLMs' inference, approximate and dynamic sparse calculate the attention, which reduces inference latency by up to 10x for pre-filling on an A100 while maintaining accuracy.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    keywords="LLMs Inference, Long-Context LLMs, Dynamic Sparse Attention, Efficient Inference",
    license="MIT License",
    url="https://github.com/microsoft/MInference",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "."},
    packages=find_packages(
        exclude=(
            "csrc",
            "dist",
            "examples",
            "experiments",
            "images",
            "test",
            "minference.egg-info",
        )
    ),
    extras_require={
        "dev": DEV_REQUIRES,
        "quality": QUANLITY_REQUIRES,
    },
    install_requires=INSTALL_REQUIRES,
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
    include_package_data=True,
    python_requires=">=3.8.0",
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
