import numpy as np
from argparse import ArgumentParser
import logging
from datetime import datetime
from time import perf_counter, sleep
import psutil

from adios2 import Stream, Adios, bindings

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import torch
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError as e:
    pass

from utils.logger import MPIFileHandler
from utils.sim_utils import setup_problem, simulation_step, generate_training_data
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
    parser.add_argument('--problem_size', default='small', type=str, choices=['small','medium','large'], help='Size of science problem to set up')
    parser.add_argument('--ppn', default=1, type=int, help='Number of MPI processes per node')
    parser.add_argument('--logging', default='info', type=str, choices=['debug', 'info'], help='Level of logging')
    parser.add_argument('--workflow_steps', type=int, default=2, help='Number of workflow steps to execute')
    parser.add_argument('--simulation_steps', type=int, default=2, help='Number of simulation steps to execute between training data transfers')
    parser.add_argument('--simulation_device', default='cuda', type=str, choices=['cpu', 'xpu', 'cuda'], help='Device to run simulation (GMRES) on')
    parser.add_argument('--inference_device', default='cuda', type=str, choices=['cpu', 'xpu', 'cuda'], help='Device to run inference on')
    parser.add_argument('--inference_precision', default='fp32', type=str, choices=['fp32', 'tf32', 'fp64', 'fp16', 'bf16'], help='Data precision used for inference')
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

    # Set device to run on
    torch.set_num_threads(1)
    gpu_device = None
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda')
        cuda_id = rankl if torch.cuda.device_count()>1 else 0
        assert cuda_id>=0 and cuda_id<torch.cuda.device_count(), \
                   f"Assert failed: cuda_id={cuda_id} and {torch.cuda.device_count()} available devices"
        torch.cuda.set_device(cuda_id)
    elif torch.xpu.is_available():
        gpu_device = torch.device('xpu')
        xpu_id = rankl if torch.xpu.device_count()>1 else 0
        assert xpu_id>=0 and xpu_id<torch.xpu.device_count(), \
                   f"Assert failed: xpu_id={xpu_id} and {torch.xpu.device_count()} available devices"
        torch.xpu.set_device(xpu_id)
    if (rank == 0):
        logger.info(f"\nFound GPU device: {gpu_device}\n")
    sim_device = gpu_device if (args.simulation_device != 'cpu') else torch.device('cpu')
    infer_device = gpu_device if (args.inference_device != 'cpu') else torch.device('cpu')

    # Instantiate and setup the model
    model = GNN(args, problem_def['n_features'], problem_def['spatial_dim'])
    model.setup_local_graph(problem_def['coords'], problem_def['edge_index'])
    model_state = model.state_dict()
    if (args.inference_precision == "fp32" or args.inference_precision == "tf32"): model.float(); dtype=torch.float32
    elif (args.inference_precision == "fp64"): model.double(); dtype=torch.float64
    elif (args.inference_precision == "fp16"): model.half(); dtype=torch.float16
    elif (args.inference_precision == "bf16"): model.bfloat16(); dtype=torch.bfloat16
    model.to(infer_device)
    
    # Loop over workflow steps
    step = 0
    timers = {
        'workflow': [],
        'simulation': [],
        'simulation_step': [],
        'inference': [],
        'data_send': [],
        'model_receive': []
    }
    if rank==0: logger.info('\nStarting loop over workflow steps')
    for istep_w in range(args.workflow_steps):
        tic_w = perf_counter()
        if rank==0: logger.info(f'Step {istep_w}')
        
         # Imitating simulation steps
        tic_s = perf_counter()
        for istep_s in range(args.simulation_steps):
            tic_s_s = perf_counter()
            x, res = simulation_step(problem_def['n_nodes_gmres'], sim_device)
            timers['simulation_step'].append(perf_counter() - tic_s_s)
            if rank==0: logger.info(f'\tSim. time step {step} in {timers["simulation_step"][-1]}')
            step+=1
        comm.Barrier()
        timers['simulation'].append(perf_counter() - tic_s)
        train_data = generate_training_data(step, args.problem_size, problem_def['coords'])

        # Send training data
        tic_d = perf_counter()
        with Stream(io, 'train_data', 'w', comm) as stream:
            stream.begin_step()
            arr_size = train_data.size
            stream.write('train_data', train_data, [size*arr_size], [rank*arr_size], [arr_size])
            stream.end_step()
        timers['data_send'].append(perf_counter() - tic_d)
        comm.Barrier()
        if rank==0: logger.info('\tSent training data')

        # Read model checkpoint looping over model parameters
        #tic_m = perf_counter()
        #with Stream(io, 'model', 'r', comm) as stream:
        #    stream.begin_step()
        #    for name, param in model.named_parameters():
        #        count = stream.inquire_variable(name).shape()[0]
        #        weights = torch.from_numpy(stream.read(name, [0], [count])).type(dtype)
        #        weights_shape = stream.read_attribute(name+'/shape')
        #        if len(weights_shape)>1: weights = weights.reshape(tuple(weights_shape))
        #        with torch.no_grad():
        #            param.data = weights.to(infer_device)
        #    stream.end_step()
        #timers['model_receive'].append(perf_counter() - tic_m)
        #comm.Barrier()
        #if rank==0: logger.info('\tRead model checkpoint')

        # Read model checkpoint on rank 0 from state dict and broadcast
        tic_m = perf_counter()
        with Stream(io, 'model', 'r', comm) as stream:
            stream.begin_step()
            if rank==0:
                for name in model_state.keys():
                    count = stream.inquire_variable(name).shape()[0]
                    weights = torch.from_numpy(stream.read(name, [0], [count])).type(dtype)
                    weights_shape = stream.read_attribute(name+'/shape')
                    if len(weights_shape)>1: weights = weights.reshape(tuple(weights_shape))
                    with torch.no_grad():
                        model_state[name] = weights
            stream.end_step()
        model_state = comm.bcast(model_state, root=0)
        model.load_state_dict(model_state, strict=True)
        timers['model_receive'].append(perf_counter() - tic_m)
        comm.Barrier()
        if rank==0: logger.info('\tRead model checkpoint')
        
        # Perform inference
        tic_i = perf_counter()
        model.eval()
        if istep_w==0:
            pos = torch.from_numpy(problem_def['coords']).type(dtype).to(infer_device)
            ei = torch.from_numpy(problem_def['edge_index']).type(torch.int64).to(infer_device)
        inputs = torch.from_numpy(train_data[:,:problem_def['n_features']]).type(dtype).to(infer_device)
        if inputs.ndim<2: inputs = inputs.reshape(-1,1)
        outputs = torch.from_numpy(train_data[:,problem_def['n_features']:]).type(dtype).to(infer_device)
        if outputs.ndim<2: outputs = outputs.reshape(-1,1)
        with torch.no_grad():
            prediction = model(inputs, ei, pos)
            local_error = model.acc_fn(prediction, outputs)
        global_avg_error = comm.allreduce(local_error.cpu()) / size
        comm.Barrier()
        timers['inference'].append(perf_counter() - tic_i)
        if rank==0: logger.info(f'\tPerformed inference with global error: {global_avg_error:>4e}')

        # Debug
        #print(istep_w,'simulation rank ',rank,' : ',torch.sum(model(torch.ones((problem_def['n_nodes'],problem_def['n_features']),dtype=dtype,device=infer_device),ei,pos)),flush=True)

        # Print workflow step time
        time_w = perf_counter() - tic_w
        timers['workflow'].append(time_w)
        comm.Barrier()
        if rank==0: logger.info(f'\tWorkflow step time [sec]: {time_w:>4e}')

    # Average time data across steps
    timers_avg = {}
    if rank==0:
        logger.info(f'\nMetrics averaged across workflow steps:')
        for key, val in timers.items():
            if len(val)>2: val.pop(0)
            avg = sum(val)/len(val)
            timers_avg[key] = avg
            logger.info(f'{key} [sec]: {avg:>4e}')
        
    # Print FOM
    if rank==0:
        logger.info(f'\nFOM:')
        # FOM 1
        ml_problem_size = size * problem_def['n_nodes'] * problem_def['n_features'] / 1.0e6
        fom_problem = ml_problem_size
        fom_time = timers_avg['workflow']
        aurora_workflow_steps = args.workflow_steps
        fom_steps = 1 + 0.1 * (args.workflow_steps - aurora_workflow_steps) / aurora_workflow_steps
        fom_1 = fom_problem * fom_steps / fom_time
        logger.info(f'Global workflow FOM: {fom_1:>4e}')

        # FOM 2
        gmres_problem_size = size * problem_def['n_nodes_gmres'] / 1.0e6
        fom_simulation = gmres_problem_size / timers_avg['simulation_step']
        n_bytes = np.array([0], dtype=np.float64).itemsize
        send_problem_size_GB = size * problem_def['n_nodes'] * problem_def['n_features'] * 2 * n_bytes / 1024**3
        fom_data_send = send_problem_size_GB  / timers_avg['data_send']
        fom_inference = ml_problem_size / timers_avg['inference']
        logger.info(f'Simulation FOM: {fom_simulation:>4e}')
        logger.info(f'Training Data Send FOM:  {fom_data_send:>4e}')
        logger.info(f'Inference FOM: {fom_inference:>4e}')

    # Finalize MPI
    mh.close()
    MPI.Finalize()


if __name__ == "__main__":
    main()


