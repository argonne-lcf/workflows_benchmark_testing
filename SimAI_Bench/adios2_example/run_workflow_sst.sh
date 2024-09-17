#!/bin/bash

PROCS=2

mpiexec -n $PROCS python simulation.py &
mpiexec -n $PROCS python train.py



