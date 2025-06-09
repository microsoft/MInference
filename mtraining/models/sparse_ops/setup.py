import os
import shutil
from setuptools import setup, find_packages


setup_dir_path = os.path.dirname(__file__)
mtraining_path = os.path.dirname(os.path.dirname(setup_dir_path))
setup_dir_path = os.path.join(setup_dir_path, "mtraining_sparse_ops")
cfg_dir_path = os.path.join(mtraining_path, "ops", "minfer", "configs")
op_dir_path = os.path.join(mtraining_path, "ops", "ring_attn", "core")

shutil.copytree(cfg_dir_path, os.path.join(setup_dir_path, "configs"), dirs_exist_ok=True)

with open(os.path.join(op_dir_path, "minference_sparse_index.py"), "r") as f:
    index_code = f.read()
with open(os.path.join(setup_dir_path, "minference_sparse_index.py"), "w") as f:
    f.write(index_code)
with open(os.path.join(op_dir_path, "minference_attn.py"), "r") as f:
    attn_code = f.read()
with open(os.path.join(setup_dir_path, "minference_attn.py"), "w") as f:
    f.write(attn_code.replace("MTraining.ops.ring_attn.core", "mtraining_sparse_ops"))

setup(
    name="mtraining_sparse_ops",  # Name of your project
    version="0.1.0",
    packages=find_packages(),  # Automatically discover all packages
)
