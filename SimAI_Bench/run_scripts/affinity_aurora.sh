#!/bin/bash

num_gpus=$1
offset=$2
component=$3
shift && shift && shift

# Get the RankID from different launcher
if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_LRANKID=$MPI_LOCALRANKID
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_LRANKID=$PALS_LOCAL_RANKID
  _MPI_RANKID=$PALS_RANKID
else
  display_help
fi

gpu=$((_MPI_LRANKID % num_gpus + offset))

unset EnableWalkerPartition
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=$gpu
#echo ?${component} RANK= ${_MPI_RANKID} LOCAL_RANK= ${_MPI_LRANKID} gpu= ${gpu}?
exec "$@"
