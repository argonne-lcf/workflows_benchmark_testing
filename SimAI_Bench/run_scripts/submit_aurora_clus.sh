#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N simai
#PBS -l walltime=00:30:00
#PBS -l select=2
#PBS -k doe
#PBS -j oe
#PBS -A Aurora_deployment
#PBS -q lustre_scaling
##PBS -m be
##PBS -M rbalin@anl.gov

cd $PBS_O_WORKDIR

# Load modules
module load frameworks/2024.2.1_u1
source /flare/Aurora_deployment/balin/ALCF-4/env_2024.2.1/_pyg/bin/activate
export PYTHONPATH=$PYTHONPATH:/flare/Aurora_deployment/balin/ALCF-4/build/adios2-build-2/lib/python3.10/site-packages/
echo Loaded modules:
module list

# Set env variables
#export SstVerbose=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Set up run
DEPLOYMENT="clustered"
NODES=$(cat $PBS_NODEFILE | wc -l)
SIM_NODES=$(( $NODES / 2 ))
TRAIN_NODES=$(( $NODES / 2 ))
SIM_PROCS_PER_NODE=12
SIM_PROCS=$((SIM_NODES * SIM_PROCS_PER_NODE))
TRAIN_PROCS_PER_NODE=12
TRAIN_PROCS=$((TRAIN_NODES * TRAIN_PROCS_PER_NODE))
echo Number of total nodes: $NODES
echo Number of simulation nodes: $SIM_NODES
echo Number of simulation ranks: $SIM_PROCS
echo Number of simulation ranks per node: $SIM_PROCS_PER_NODE
echo Number of trainer nodes: $TRAIN_NODES
echo Number of trainer ranks: $TRAIN_PROCS
echo Number of trainer ranks per node: $TRAIN_PROCS_PER_NODE
echo

# Parse node list
if [ $DEPLOYMENT = "colocated" ]; then
    SIM_HOSTFILE=$PBS_NODEFILE
    TRAIN_HOSTFILE=$PBS_NODEFILE
    SIM_CPU_BIND="list:1:8:16:24:32:40"
    TRAIN_CPU_BIND="list:53:60:68:76:84:92"
elif [ $DEPLOYMENT = "clustered" ]; then
    if [ "$(( $NODES % 2 ))" -eq 0 ]; then
        NLINES=$SIM_NODES
        NLINESP1=$(( $NLINES + 1 ))
        sed -n 1,${NLINES}p $PBS_NODEFILE > ./sim_hostfile
        sed -n ${NLINESP1},${NODES}p $PBS_NODEFILE > ./train_hostfile
        SIM_HOSTFILE="./sim_hostfile"
        TRAIN_HOSTFILE="./train_hostfile"
        SIM_CPU_BIND="list:1:8:16:24:32:40:53:60:68:76:84:92"
        TRAIN_CPU_BIND="list:1:8:16:24:32:40:53:60:68:76:84:92"
    else
        echo "Currently need an even amount of total nodes for clustered deployment"
        exit 1
    fi
fi
echo Simulation running on:
echo `cat $SIM_HOSTFILE`
echo
echo Training running on:
echo `cat $TRAIN_HOSTFILE`
echo

# Workflow parameters
PROBLEM="medium"
STEPS=20
SIM_STEPS=3
TRAIN_STEPS=5
DEVICE="xpu"
LOGGING="debug"
LAYER="WAN"

# Run
EXE_PATH=/flare/Aurora_deployment/balin/ALCF-4/workflows_benchmark_testing/SimAI_Bench/src
AFFINITY_PATH=/flare/Aurora_deployment/balin/ALCF-4/workflows_benchmark_testing/SimAI_Bench/run_scripts

if ls *.sst 1> /dev/null 2>&1
then 
    echo Cleaning up old .sst files
    rm *.sst
fi

echo Running workflow ...
echo `date`
mpiexec --pmi=pmix --envall --hostfile $SIM_HOSTFILE -n $SIM_PROCS --ppn $SIM_PROCS_PER_NODE \
    --cpu-bind=$SIM_CPU_BIND $AFFINITY_PATH/affinity_aurora.sh $SIM_PROCS_PER_NODE 0 "SIM" \
    python $EXE_PATH/simulation.py \
    --ppn $SIM_PROCS_PER_NODE \
    --problem_size $PROBLEM \
    --workflow_steps $STEPS \
    --simulation_steps $SIM_STEPS \
    --simulation_device $DEVICE \
    --inference_device $DEVICE \
    --logging $LOGGING \
    --adios_transport $LAYER &

mpiexec --pmi=pmix --envall --hostfile $TRAIN_HOSTFILE -n $TRAIN_PROCS --ppn $TRAIN_PROCS_PER_NODE \
    --cpu-bind=$TRAIN_CPU_BIND $AFFINITY_PATH/affinity_aurora.sh $TRAIN_PROCS_PER_NODE 0 "TRAIN" \
    python $EXE_PATH/trainer.py \
    --ppn $TRAIN_PROCS_PER_NODE \
    --workflow_steps $STEPS \
    --training_iters $TRAIN_STEPS \
    --device $DEVICE \
    --logging $LOGGING \
    --adios_transport $LAYER
wait
echo `date`

# Clean up
JOBID=$(echo $PBS_JOBID | awk '{split($1,a,"."); print a[1]}')
mkdir $JOBID
mv sim_* train_* $JOBID
