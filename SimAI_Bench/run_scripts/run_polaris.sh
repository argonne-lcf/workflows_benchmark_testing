#!/bin/bash

# Load modules
module use /soft/modulefiles
module load spack-pe-gnu
module load adios2/2.10.0-cuda
module load cudatoolkit-standalone/12.4.0
source /eagle/datascience/balin/ALCF-4/venv/_adios2/bin/activate
echo Loaded modules:
module list

# Set env variables
EXE_PATH=/eagle/datascience/balin/ALCF-4/workflows_benchmark_testing/SimAI_Bench/src
#export SstVerbose=1

# Set up run
NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=2
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of simulation ranks: $PROCS
echo Number of simulation ranks per node: $PROCS_PER_NODE
echo Number of trainer ranks: $PROCS
echo Number of trainer ranks per node: $PROCS_PER_NODE
echo

echo Running workflow ...
echo `date`
mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1:2:3:4 python $EXE_PATH/simulation.py --ppn $PROCS_PER_NODE &
mpiexec -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:7:8:9:10 python $EXE_PATH/trainer.py --ppn $PROCS_PER_NODE
wait
echo `date`


