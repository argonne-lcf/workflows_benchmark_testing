#!/bin/bash -l
#PBS -l select=1024
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q lustre_scaling
#PBS -A Aurora_deployment
#PBS -k doe
#PBS -N mpi_ensemble_test

cd ${PBS_O_WORKDIR}

EXE="/home/csimpson/benchmark_testing/hello_sleeper.sh 300"
launch_iter=1

echo $EXE
hostname

# Set output directory
if ! test -f ./batch_iter; then
    echo -1 > ./batch_iter
fi

BATCH_NUM=$(cat ./batch_iter)
BATCH_NUM=$(( BATCH_NUM + 1 ))
echo ${BATCH_NUM} > batch_iter
echo iter=${BATCH_NUM}
OUTDIR="./sleep_test"${BATCH_NUM}
echo "OUTDIR= ${OUTDIR}"
mkdir ${OUTDIR}

# MPI example w/ multiple runs per batch job
cp ${PBS_NODEFILE} ${OUTDIR}/pbs_nodefile
NNODES=`wc -l < $PBS_NODEFILE`
NODE_LIST=`cat $PBS_NODEFILE`
num_gpu=$(/usr/bin/udevadm info /sys/module/i915/drivers/pci:i915/* |& grep -v Unknown | grep -c "P: /devices")
num_tile=2

# User must ensure there are enough nodes in job to support all concurrent runs
NUM_NODES_PER_MPI=1
NODE_PACKING_COUNT=12
NRANKS_PER_NODE=1
NDEPTH=8
NTHREADS=1
NTOTRANKS=$(( NUM_NODES_PER_MPI * NRANKS_PER_NODE ))

echo "NUM_OF_NODES=${NNODES} NUM_NODES_PER_MPI=${NUM_NODES_PER_MPI} TOTAL_NUM_RANKS=${NTOTRANKS} RANKS_PER_NODE=${NRANKS_PER_NODE} THREADS_PER_RANK=${NTHREADS} NODE_PACKING_COUNT=${NODE_PACKING_COUNT}"

# Increase value of suffix-length if more than 99 jobs
#split --lines=${NUM_NODES_PER_MPI} --numeric-suffixes=1 --suffix-length=3 $PBS_NODEFILE ${OUTDIR}/local_hostfile.
while [[ ${launch_iter} -ge 1 ]]
do
    iter=0
    node_iter=0
    SECONDS=0
    #for lh in ${OUTDIR}/local_hostfile*
    for hn in $NODE_LIST
    do
	nmpi_per_node=$(( NODE_PACKING_COUNT - 1 ))
	while [[ ${nmpi_per_node} -ge 0 ]]
	do
	    gpu_id=$(((nmpi_per_node / num_tile) % num_gpu))
	    tile_id=$((nmpi_per_node % num_tile))
	    thread_id=$(((NODE_PACKING_COUNT - nmpi_per_node - 1) * NDEPTH ))

	    #echo "Launching mpiexec on ${hn}"
	    mkdir ${OUTDIR}/${iter}
	
	    #ZE_AFFINITY_MASK=${gpu_id}.${tile_id} mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --hostfile ${lh} --cpu-bind list:${thread_id} ${EXE} > ${OUTDIR}/${iter}/job.out &

	    #hn=`cat $lh`
	    ZE_AFFINITY_MASK=${gpu_id}.${tile_id} mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --hosts ${hn} --cpu-bind list:${thread_id} ${EXE} >> ${OUTDIR}/${iter}/job.out &
	
	    iter=$(( iter + 1))
	    nmpi_per_node=$(( nmpi_per_node - 1 ))
	done
	node_iter=$(( node_iter + 1 ))
    done
    echo "Launch time is ${SECONDS} seconds for ${iter} mpi calls on ${node_iter} nodes"

    wait
    launch_iter=$(( launch_iter - 1 ))
done
