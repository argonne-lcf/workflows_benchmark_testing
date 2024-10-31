#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N simai
#PBS -l walltime=00:30:00
#PBS -l select=1
#PBS -k doe
#PBS -j oe
#PBS -A Aurora_deployment
#PBS -q lustre_scaling
#PBS -V
##PBS -m be
##PBS -M rbalin@anl.gov

cd $PBS_O_WORKDIR

# Load modules
module load frameworks/2024.2.1_u1
source /flare/Aurora_deployment/balin/ALCF-4/env_2024.2.1/_pyg/bin/activate
export PYTHONPATH=$PYTHONPATH:/opt/aurora/24.180.1/spack/unified/0.8.0/install/linux-sles15-x86_64/oneapi-2024.07.30.002/adios2-2.10.0-z7daajo/venv-1.0-zrqfop4/lib/python3.10/site-packages/
echo Loaded modules:
module list

# Set env variables
#export SstVerbose=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Set up run
NODES=$(cat $PBS_NODEFILE | wc -l)
PROCS_PER_NODE=6
PROCS=$((NODES * PROCS_PER_NODE))
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
echo Number of nodes: $NODES
echo Number of simulation ranks: $PROCS
echo Number of simulation ranks per node: $PROCS_PER_NODE
echo Number of trainer ranks: $PROCS
echo Number of trainer ranks per node: $PROCS_PER_NODE
echo

if ls *.sst 1> /dev/null 2>&1
then 
    echo Cleaning up old .sst files
    rm *.sst
fi

# Workflow parameters
PROBLEM="medium"
STEPS=2
SIM_STEPS=10
TRAIN_STEPS=10
DEVICE="xpu"

# Run
EXE_PATH=/flare/Aurora_deployment/balin/ALCF-4/workflows_benchmark_testing/SimAI_Bench/src
echo Running workflow ...
echo `date`
#mpiexec --pmi=pmix --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1:8:16:24:32:40 ./affinity_aurora.sh $PROCS_PER_NODE 0 \
mpiexec --pmi=pmix --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:1:8:16:24:32:40 \
    python $EXE_PATH/simulation.py \
    --ppn $PROCS_PER_NODE \
    --problem_size $PROBLEM \
    --workflow_steps $STEPS \
    --simulation_steps $SIM_STEPS \
    --simulation_device $DEVICE \
    --inference_device $DEVICE &
#mpiexec --pmi=pmix --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:53:60:68:76:84:92 ./affinity_aurora.sh $PROCS_PER_NODE $PROCS_PER_NODE \
mpiexec --pmi=pmix --envall -n $PROCS --ppn $PROCS_PER_NODE --cpu-bind=list:53:60:68:76:84:92 \
    python $EXE_PATH/trainer.py \
    --ppn $PROCS_PER_NODE \
    --workflow_steps $STEPS \
    --training_iters $TRAIN_STEPS \
    --device $DEVICE
wait
echo `date`


