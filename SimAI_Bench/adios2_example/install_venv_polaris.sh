#!/bin/bash

WDIR=$PWD

# Load modules
module use /soft/modulefiles
module load spack-pe-gnu
module load adios2/2.10.0-cuda
#module load conda/2024-04-29

# Create venv
python3 -m venv --clear _adios2 --system-site-packages
. _adios2/bin/activate

pip install matplotlib


