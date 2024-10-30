#!/bin/bash

num_gpus=$1
offset=$2
shift
shift

_MPI_LRANKID=${PMI_LOCAL_RANK}
gpu=$((_MPI_LRANKID % num_gpus + offset))

export CUDA_VISIBLE_DEVICES=$gpu
#echo ?RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}?
exec "$@"
