#!/bin/bash

PROCS=2
PPN=1

# Load modules
module use /soft/modulefiles
module load spack-pe-gnu
module load adios2/2.10.0-cuda
source /eagle/datascience/balin/ALCF-4/venv/_adios2/bin/activate

#export SstVerbose=1

mpiexec -n $PROCS python simulation.py &
mpiexec -n $PROCS python train.py



