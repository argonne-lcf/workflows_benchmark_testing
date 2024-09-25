import numpy as np
from time import sleep
from argparse import ArgumentParser
import logging
from datetime import datetime
import psutil

from adios2 import Stream, Adios, bindings

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

from utils.logger import MPIFileHandler
from utils.sim_utils import setup_problem, simulation_step

# Main simulation function
def main():
    """Emulate a data producing simulation with online training and inference
    """

    # MPI Init
    MPI.Init()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    comm.Barrier()

    # Parse arguments
    parser = ArgumentParser(description='SimAI-Bench Simulation')
    parser.add_argument('--problem_size', default='small', type=str, choices=['small'], help='Size of science problem to set up')
    parser.add_argument('--ppn', default=1, type=int, help='Number of MPI processes per node')
    parser.add_argument('--logging', default='debug', type=str, choices=['debug', 'info'], help='Level of logging')
    parser.add_argument('--simulation_steps', type=int, default=5, help='Number of simulation steps to execute between training data transfers')
    parser.add_argument('--workflow_steps', type=int, default=2, help='Number of workflow steps to execute')
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
    mh = MPIFileHandler(f"sim_{date}.log", comm=comm)
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
    if rank==0:
        logger.info(f'\nRunning {args.problem_size} problem size, ')
        logger.info(f'with {args.adios_engine} ADIOS2 engine \n')

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

    # Setup problem and send definition to trainer
    problem_def = setup_problem(args,comm)
    with Stream(io, 'problem_definition', 'w', comm) as stream:
        stream.begin_step()
        if rank == 0:
            stream.write('n_nodes',problem_def['n_nodes'])
            stream.write('n_edges',problem_def['n_edges'])
            stream.write('n_features',problem_def['n_features'])
            stream.write('n_targets',problem_def['n_targets'])
            stream.write('spatial_dim',problem_def['spatial_dim'])
        arr_size = problem_def['coords'].size
        stream.write('coords', problem_def['coords'].flatten(), [arr_size*size], [rank*arr_size], [arr_size])
        arr_size = problem_def['edge_index'].size
        stream.write('edge_index', problem_def['edge_index'].flatten(), [arr_size*size], [rank*arr_size], [arr_size])
        stream.end_step()
    comm.Barrier()
    if rank==0: logger.info('Simulation sent problem definition')

    # Loop over workflow steps
    step = 0
    for istep_w in range(args.workflow_steps):
        # Imitating simulation steps
        for istep_s in range(args.simulation_steps):
            sleep(0.5)
            train_data = simulation_step(step, args.problem_size, problem_def['coords'])
            step+=1

        # Send training data
        with Stream(io, 'train_data', 'w', comm) as stream:
            stream.begin_step()
            arr_size = train_data.size
            stream.write('train_data', train_data, [size*arr_size], [rank*arr_size], [arr_size])
            stream.end_step()
        comm.Barrier()
        if rank==0: logger.info('Simulation sent training data')

        # Read model checkpoint
        with Stream(io, 'model', 'r', comm) as stream:
            stream.begin_step()
            model = stream.read("model")
            stream.end_step()
        comm.Barrier()
        if rank==0: logger.info('Simulation read model checkpoint')

    # Finalize MPI
    mh.close()
    MPI.Finalize()


if __name__ == "__main__":
    main()


