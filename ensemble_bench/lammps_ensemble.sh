#!/bin/bash -l
#PBS -l select=128
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q lustre_scaling
#PBS -A Aurora_deployment
#PBS -k doe
#PBS -N lammps_ensemble_test

cd ${PBS_O_WORKDIR}

export MPICH_GPU_SUPPORT_ENABLED=1

NNODES=`wc -l < $PBS_NODEFILE`
NNODES_PER_TASK=2
PPN=12
NRANKS=$(( NNODES_PER_TASK*PPN ))
NDEPTH=8
NTHREADS=8

ATOMS_PER_RANK=10700000
NATOMS=$(( ATOMS_PER_RANK*NRANKS ))
NCUBED=$(( NATOMS/4 ))
#NX=`echo "e( l($NCUBED)/3 )" | bc -l`
#NX_INT=`printf "%*.f\n" 0 $NX`
NX=400

LAMMPS=$HOME/bin/lmp_aurora_kokkos
INPUTS="-in ${HOME}/bin/lj_lammps_template.in -k on g 1 -var nx ${NX} -sf kk -pk kokkos neigh half neigh/qeq full newton on"
AFF="/soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh"

echo Launch host
hostname

# Set output directory
if ! test -f ./batch_iter; then
    echo -1 > ./batch_iter
fi

BATCH_NUM=$(cat ./batch_iter)
BATCH_NUM=$(( BATCH_NUM + 1 ))
echo ${BATCH_NUM} > batch_iter
echo iter=${BATCH_NUM}
OUTDIR=${PBS_O_WORKDIR}"/sleep_test"${BATCH_NUM}
echo "OUTDIR= ${OUTDIR}"
mkdir ${OUTDIR}

# MPI example w/ multiple runs per batch job
cp ${PBS_NODEFILE} ${OUTDIR}/pbs_nodefile
NODE_LIST=`cat $PBS_NODEFILE`

# User must ensure there are enough nodes in job to support all concurrent runs
NTASKS=$(( NNODES/NNODES_PER_TASK ))

# Increase value of suffix-length if more than 99 jobs
#split --lines=${NUM_NODES_PER_MPI} --numeric-suffixes=1 --suffix-length=3 $PBS_NODEFILE ${OUTDIR}/local_hostfile.

SECONDS=0
iter=0
#for hn in $NODE_LIST
for ((i = 0 ; i < $NTASKS ; i++ ))
do
    RUN_DIR=${OUTDIR}/${i}
    mkdir $RUN_DIR
    echo $RUN_DIR
    cd $RUN_DIR

    # Form node list
    start_line=$(( i*2+1 ))
    hn_list=`echo $NODE_LIST | cut -d " " -f $start_line`
    for ((n = 1 ; n < $NNODES_PER_TASK ; n++ ))
    do
	next_line=$(( start_line+n ))
	hn=`echo $NODE_LIST | cut -d " " -f $next_line`
	hn_list="${hn_list},${hn}"
    done
    echo $hn_list
    mpiexec -n ${NRANKS} --ppn ${PPN} --depth=${NDEPTH} --cpu-bind depth --hosts ${hn_list} --env OMP_NUM_THREADS=${NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=cores ${AFF} ${LAMMPS} ${INPUTS} >> ${RUN_DIR}/job.out &
    iter=$(( iter + 1))
done
echo "Launch time is ${SECONDS} seconds for ${iter} mpi calls on ${NNODES} nodes"
wait
echo "Total time is ${SECONDS} seconds for ${iter} mpi calls on ${NNODES} nodes"
