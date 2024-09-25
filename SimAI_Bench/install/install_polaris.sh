#!/bin/bash

WDIR=$PWD

# Load modules
module use /soft/modulefiles
module load spack-pe-gnu
module load adios2/2.10.0-cuda
module load cudatoolkit-standalone/12.4.0

# Create venv
python3 -m venv --clear _adios2 --system-site-packages
. _adios2/bin/activate

pip install matplotlib
pip install pyyaml
pip install mpipartition
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html


