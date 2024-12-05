#!/bin/bash 

module load frameworks/2024.2.1_u1

python3 -m venv --clear _pyg --system-site-packages
source _pyg/bin/activate

export LD_LIBRARY_PATH=/opt/aurora/24.180.1/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

# PyTorch Geometric and utils
pip install torch_geometric==2.5.3 # match Polaris version

# torch_scatter CPU only
#git clone https://github.com/rusty1s/pytorch_scatter.git
#cd pytorch_scatter
#git checkout 2.1.1 
#pip install .
#cd ..

# torch_cluster on CPU only
git clone https://github.com/rusty1s/pytorch_cluster.git
cd pytorch_cluster
git checkout 1.6.1
# NB: Comment lines 53-61 of setup.py to remove OpenMP backend build, which gives torch_cluster/_fps_cpu.so: undefined symbol: __kmpc_fork_call error. I tried changing the -fopenmp flag to -fiopenmp, but this was not enough
pip install .
cd ..

#Other packages
pip install mpipartition

