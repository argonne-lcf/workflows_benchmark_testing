#!/bin/bash

PROCS=4
PPN=4

# Load modules
#module load adios2/2.10.0-sycl 
#export PYTHONPATH=/opt/aurora/24.180.0/spack/unified/0.8.0/install/linux-sles15-x86_64/oneapi-2024.07.30.002/adios2-2.10.0-pexm7hbmzd7l6ml6jtfddsy66v4ndhvf/venv-1.0-zrqfop4de73b3ka53zngkbdbrk73oulz/lib/python3.10/site-packages:$PYTHONPATH

#module load adios2/2.10.0-cpu 
#export PYTHONPATH=/opt/aurora/24.180.0/spack/unified/0.8.0/install/linux-sles15-x86_64/oneapi-2024.07.30.002/adios2-2.10.0-stqtunnu5zpgc6h2fblmafko3imylepi/venv-1.0-zrqfop4de73b3ka53zngkbdbrk73oulz/lib/python3.10/site-packages:$PYTHONPATH

#export SstVerbose=1

mpiexec -n $PROCS python simulation.py &
mpiexec -n $PROCS python train.py



