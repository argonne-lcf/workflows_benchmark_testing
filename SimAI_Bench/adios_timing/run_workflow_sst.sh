#!/bin/bash

PROCS=2
PPN=1

# Load modules
#module use /soft/modulefiles
#module load spack-pe-gnu
#module load adios2/2.10.0-cuda
#source /eagle/datascience/balin/ALCF-4/venv/_adios2/bin/activate

module load frameworks/2024.2.1_u1
source /gila/Aurora_deployment/balin/ALCF-4/env_2024.2.1/_pyg/bin/activate
export PYTHONPATH=$PYTHONPATH:/gila/Aurora_deployment/balin/ALCF-4/build/adios2-build-2/install/lib/python3.10/site-packages/

export SstVerbose=2

#mpiexec -n $PROCS python simulation.py &
#mpiexec -n $PROCS python train.py
mpiexec -n $PROCS python simulation.py : -n $PROCS python train.py



