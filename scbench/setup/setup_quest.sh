# make sure cuda toolkit is installed
# conda install -c conda-forge cuda-toolkit=12.1 -y

git clone --recurse-submodules https://github.com/mit-han-lab/quest /tmp/quest
cd /tmp/quest
pip install -e . --no-deps
pip install ninja packaging

conda install cmake -y

# build libraft
cd kernels/3rdparty/raft
./build.sh libraft

# This will automatically build and link the operators
cd ../../..
cd quest/ops
bash setup.sh
