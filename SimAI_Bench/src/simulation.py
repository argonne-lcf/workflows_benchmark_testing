import numpy as np
from argparse import ArgumentParser
import logging
from datetime import datetime
from time import perf_counter
import psutil

from adios2 import Stream, Adios, bindings

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import torch

from utils.logger import MPIFileHandler
from utils.sim_utils import setup_problem, simulation_step
from gnn.model import GNN

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

    # Parse arguments
    parser = ArgumentParser(description='SimAI-Bench Simulation')
    parser.add_argument('--problem_size', default='small', type=str, choices=['small'], help='Size of science problem to set up')
    parser.add_argument('--ppn', default=1, type=int, help='Number of MPI processes per node')
    parser.add_argument('--logging', default='debug', type=str, choices=['debug', 'info'], help='Level of logging')
    parser.add_argument('--simulation_steps', type=int, default=5, help='Number of simulation steps to execute between training data transfers')
    parser.add_argument('--workflow_steps', type=int, default=2, help='Number of workflow steps to execute')
    parser.add_argument('--inference_precision', default='fp32', type=str, choices=['fp32', 'tf32', 'fp64', 'fp16', 'bf16'], help='Data precision used for inference')
    parser.add_argument('--inference_device', default='cuda', type=str, choices=['cpu', 'xpu', 'cuda'], help='Device to run inference on')
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
    comm.Barrier()
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

    # Instantiate and setup the model
    model = GNN(args, problem_def['n_features'], problem_def['spatial_dim'])
    model.setup_local_graph(problem_def['coords'], problem_def['edge_index'])
    if (args.inference_precision == "fp32" or args.inference_precision == "tf32"): model.float(); dtype=torch.float32
    elif (args.inference_precision == "fp64"): model.double(); dtype=torch.float64
    elif (args.inference_precision == "fp16"): model.half(); dtype=torch.float16
    elif (args.inference_precision == "bf16"): model.bfloat16(); dtype=torch.bfloat16
    if (args.inference_device != 'cpu'): model.to(args.inference_device)

    # Loop over workflow steps
    step = 0
    timers = {
        'workflow': [],
    }
    if rank==0: logger.info('\nStarting loop over workflow steps')
    for istep_w in range(args.workflow_steps):
        tic_w = perf_counter()
        if rank==0: logger.info(f'Step {istep_w}')
        
         # Imitating simulation steps
        for istep_s in range(args.simulation_steps):
            train_data = simulation_step(step, args.problem_size, problem_def['coords'])
            step+=1

        # Send training data
        with Stream(io, 'train_data', 'w', comm) as stream:
            stream.begin_step()
            arr_size = train_data.size
            stream.write('train_data', train_data, [size*arr_size], [rank*arr_size], [arr_size])
            stream.end_step()
        comm.Barrier()
        if rank==0: logger.info('\tSent training data')

        # Read model checkpoint
        with Stream(io, 'model', 'r', comm) as stream:
            stream.begin_step()
            for name, param in model.named_parameters():
                count = stream.inquire_variable(name).shape()[0]
                weights = torch.from_numpy(stream.read(name, [0], [count])).type(dtype)
                weights_shape = stream.read_attribute(name+'/shape')
                if len(weights_shape)>1: weights = weights.reshape(tuple(weights_shape))
                with torch.no_grad():
                    param.data = weights.to(args.inference_device)
            stream.end_step()
        comm.Barrier()
        if rank==0: logger.info('\tRead model checkpoint')

        # Perform inference
        model.eval()
        if istep_w==0:
            pos = torch.from_numpy(problem_def['coords']).type(dtype).to(args.inference_device)
            ei = torch.from_numpy(problem_def['edge_index']).type(torch.int64).to(args.inference_device)
        inputs = torch.from_numpy(train_data[:,problem_def['n_features']]).type(dtype).to(args.inference_device)
        if inputs.ndim<2: inputs = inputs.reshape(-1,1)
        outputs = torch.from_numpy(train_data[:,problem_def['n_features']:]).type(dtype).to(args.inference_device)
        if outputs.ndim<2: outputs = outputs.reshape(-1,1)
        with torch.no_grad():
            prediction = model(inputs, ei, pos)
            local_error = model.acc_fn(prediction, outputs)
        global_avg_error = comm.allreduce(local_error) / size
        comm.Barrier()
        if rank==0: logger.info(f'\tPerformed inference with global error: {global_avg_error:>4e}')

        # Debug
        #print(istep_w,'simulation rank ',rank,' : ',torch.sum(model(torch.ones((problem_def['n_nodes'],problem_def['n_features']),dtype=dtype,device=args.inference_device),ei,pos)),flush=True)

        # Print workflow step time
        time_w = perf_counter() - tic_w
        if rank==0: logger.info(f'\tWorkflow step time [sec]: {time_w:>4e}')
        timers['workflow'].append(time_w)

    # Average time data across steps
    if rank==0:
        logger.info(f'\nMetrics averaged across workflow steps:')
        for key, val in timers.items():
            if len(val)>2: val.pop(0)
            avg = sum(val)/len(val)
            logger.info(f'{key} [sec]: {avg:>4e}')
        
    # Print FOM
    if rank==0:
        logger.info(f'\nFOM:')
        fom_problem = size * problem_def['n_nodes'] * (problem_def['n_features'] * 2)
        fom_time = sum(timers['workflow'])/len(timers['workflow'])
        aurora_workflow_steps = args.workflow_steps
        fom_steps = 1 + 0.1 * (args.workflow_steps - aurora_workflow_steps) / aurora_workflow_steps
        fom_1 = fom_problem * fom_steps / fom_time
        logger.info(f'FOM 1: {fom_1:>4e}')
         

    # Finalize MPI
    mh.close()
    MPI.Finalize()


if __name__ == "__main__":
    main()


