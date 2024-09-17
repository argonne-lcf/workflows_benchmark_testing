#!/bin/bash

PROCS=2

mpiexec -n $PROCS python simulation.py : -n $PROCS python train.py



