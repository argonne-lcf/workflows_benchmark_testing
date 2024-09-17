#!/bin/bash

PROCS=4
PPN=2

# Load modules
module use /soft/modulefiles
module load spack-pe-gnu
module load adios2/2.10.0-cuda
source /eagle/datascience/balin/ALCF-4/venv/_adios2/bin/activate

mpiexec -n $PROCS python simulation.py &
mpiexec -n $PROCS python train.py



