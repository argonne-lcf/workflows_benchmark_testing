import os
import socket
import numpy as np
from time import sleep
import logging
from argparse import ArgumentParser
from datetime import datetime, timedelta
import psutil

from adios2 import Stream, Adios, bindings

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import torch
import torch.distributed as dist

from utils.logger import MPIFileHandler
from gnn.model import GNN
import utils.train_utils as utils


# Main trainer function
def main():
    """Perform training of the GNN model
    """

    # MPI Init
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # Parse arguments
    parser = ArgumentParser(description='SimAI-Bench Trainer')
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'xpu', 'cuda'], help='Device to run training on')
    parser.add_argument('--ppn', default=1, type=int, help='Number of MPI processes per node')
    parser.add_argument('--logging', default='debug', type=str, choices=['debug', 'info'], help='Level of logging')
    parser.add_argument('--workflow_steps', type=int, default=2, help='Number of workflow steps to execute')
    parser.add_argument('--training_steps', type=int, default=5, help='Number of training steps to execute per workflow step')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Number of hidden node features in GNN')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='Number of hidden layers for encoder/decoder, edge update, node update layers MLPs')
    parser.add_argument('--message_passing_layers', type=int, default=4, help='Number of GNN message pssing layers')
    parser.add_argument('--adios_engine', type=str, default='SST', choices=['SST'], help='ADIOS2 transport engine')
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.logging.upper())
    logger = logging.getLogger(f'[{rank}]')
    logger.setLevel(log_level)
    date = datetime.now().strftime('%d.%m.%y_%H.%M') if rank==0 else None
    date = comm.bcast(date, root=0)
    comm.Barrier()
    #formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    formatter = logging.Formatter('%(message)s')
    mh = MPIFileHandler(f"train_{date}.log", comm=comm)
    mh.setFormatter(formatter)
    logger.addHandler(mh)

    # Print setup information
    rankl = rank % args.ppn
    if args.logging=='debug':
        try:
            p = psutil.Process()
            core_list = p.cpu_affinity()
        except:
            core_list = []
        logger.debug(f"Hello from MPI rank {rank}/{size}, local rank {rankl}, " \
                     +f"cores {core_list}, and node {name}")

    # Intel imports
    try:
        import intel_extension_for_pytorch
    except ModuleNotFoundError as e:
        if rank==0: logger.warning(f'{e}')
    try:
        import oneccl_bindings_for_pytorch
        ONECCL = True
    except ModuleNotFoundError as e:
        ONECCL = False
        if rank==0: logger.warning(f'{e}')

    # Initialize Torch Distributed
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(size)
    master_addr = socket.gethostname() if rank == 0 else None
    master_addr = comm.bcast(master_addr, root=0)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(2345)
    if (args.device=='cuda'): backend = 'nccl'
    elif (args.device=='xpu' or ONECCL): backend = 'ccl'
    elif (args.device=='cpu'): backend = 'gloo'
    dist.init_process_group(backend,
                            rank=int(rank),
                            world_size=int(size),
                            init_method='env://',
                            timeout=timedelta(seconds=120))

    # Initialize ADIOS MPI Communicator
    adios = Adios(comm)
    io = adios.declare_io('SimAIBench')
    io.set_engine(args.adios_engine)
    parameters = {
        'RendezvousReaderCount': '1', # options: 1 for sync, 0 for async
        'QueueFullPolicy': 'Block', # options: Block, Discard
        'QueueLimit': '1', # options: 0 for no limit
        'DataTransport': 'WAN', # options: MPI, WAN,  UCX, RDMA
        'OpenTimeoutSecs': '600', # number of seconds SST is to wait for a peer connection on Open()
    }
    io.set_parameters(parameters)

    # Read problem definition data
    with Stream(io, 'problem_definition', 'r', comm) as stream:
        stream.begin_step()
        
        n_nodes = stream.read('n_nodes')
        n_edges = stream.read('n_edges')
        n_features = stream.read('n_features')
        n_targets = stream.read('n_targets')
        spatial_dim = stream.read('spatial_dim')
        
        arr = stream.inquire_variable('coords')
        shape = arr.shape()
        count = int(shape[0] / size)
        start = count * rank
        if rank == size - 1:
            count += shape[0] % size
        coords = stream.read('coords', [start], [count]).reshape((n_nodes,spatial_dim))
        assert coords.shape[0] == n_nodes and coords.shape[1] == spatial_dim       
 
        arr = stream.inquire_variable('edge_index')
        shape = arr.shape()
        count = int(shape[0] / size)
        start = count * rank
        if rank == size - 1:
            count += shape[0] % size
        edge_index = stream.read('edge_index', [start], [count]).reshape((2,n_edges))
        assert edge_index.shape[0] == 2 and edge_index.shape[1] == n_edges
        
        stream.end_step()
    comm.Barrier()
    if rank==0: logger.info('\nTrainer read problem definition')

    # Set device to run on
    torch.set_num_threads(1)
    if (args.device == 'cuda'):
        if torch.cuda.is_available():
            device = torch.device(args.device)
            cuda_id = rankl if torch.cuda.device_count()>1 else 0
            assert cuda_id>=0 and cuda_id<torch.cuda.device_count(), \
                   f"Assert failed: cuda_id={cuda_id} and {torch.cuda.device_count()} available devices"
            torch.cuda.set_device(cuda_id)
        else:
            device = torch.device('cpu')
            logger.warning(f"[{rank}]: no cuda devices available, cuda.device_count={torch.cuda.device_count()}")
    elif (args.device=='xpu'):
        if torch.xpu.is_available():
            device = torch.device(args.device)
            xpu_id = rankl if torch.xpu.device_count()>1 else 0
            assert xpu_id>=0 and xpu_id<torch.xpu.device_count(), \
                   f"Assert failed: xpu_id={xpu_id} and {torch.xpu.device_count()} available devices"
            torch.xpu.set_device(xpu_id)
        else:
            device = torch.device('cpu')
            logger.warning(f"[{rank}]: no XPU devices available, xpu.device_count={torch.xpu.device_count()}")
    if (rank == 0):
        logger.info(f"\nRunning on device: {device.type}\n")

    # Instantiate the model
    model = GNN(args, n_features, spatial_dim)
    model.setup_local_graph(coords, edge_index)
    n_params = utils.count_weights(model)
    if (rank == 0):
        logger.info(f"Loaded model with {n_params} trainable parameters \n")
    if (device.type != 'cpu'):
        model.to(device)

    # Loop over workflow steps
    if rank==0: logger.info('\nStarting loop over workflow steps')
    for istep_w in range(args.workflow_steps):
        if rank==0: logger.info(f'Step {istep_w}')

        # Read training data
        with Stream(io, "train_data", "r", comm) as stream:
            stream.begin_step()    
            arr = stream.inquire_variable('train_data')
            shape = arr.shape()
            count = int(shape[0] / size)
            start = count * rank
            if rank == size - 1:
                count += shape[0] % size
            train_data = stream.read('train_data', [start], [count])
            stream.end_step()
        comm.Barrier()
        if rank==0: logger.info('\tRead training data')

        # Imitating training steps
        sleep(5.0)

        # Send trained model back
        with Stream(io, 'model', 'w', comm) as stream:
            stream.begin_step()
            if rank == 0:
                stream.write('model', istep_w)
            stream.end_step()
        comm.Barrier()
        if rank==0: logger.info('\tSent model')

    # Finalize MPI
    dist.destroy_process_group()
    mh.close()
    MPI.Finalize()

if __name__ == "__main__":
    main()
