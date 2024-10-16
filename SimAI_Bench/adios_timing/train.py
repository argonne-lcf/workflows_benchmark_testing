from mpi4py import MPI
import numpy as np
from adios2 import Stream, Adios, bindings
from time import sleep, perf_counter

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ADIOS MPI Communicator
adios = Adios(comm)

# ADIOS IO
io = adios.declare_io("simai")
io.set_engine("SST")
parameters = {
    'RendezvousReaderCount': '1', # options: 1 for sync, 0 for async
    'QueueFullPolicy': 'Block', # options: Block, Discard
    'QueueLimit': '1', # options: 0 for no limit
    'DataTransport': 'WAN', # options: MPI, WAN,  UCX, RDMA
    'OpenTimeoutSecs': '600', # number of seconds SST is to wait for a peer connection on Open()
}
io.set_parameters(parameters)

# Read data
tic = perf_counter()
with Stream(io, "data", "r", comm) as stream:
    stream.begin_step()
    var = stream.inquire_variable("y")
    shape = var.shape()
    count = int(shape[0] / size)
    start = count * rank
    if rank == size - 1:
        count += shape[0] % size
    ticc = perf_counter()
    train_data = stream.read("y", [start], [count])
    print(f'[T{rank}] read time: {perf_counter()-ticc:>4e}',flush=True)
    ticc = perf_counter()
    stream.end_step()
    print(f'[T{rank}] end_step time: {perf_counter()-ticc:>4e}',flush=True)
    sleep(5)
print(f'[T{rank}] total read time: {perf_counter()-tic:>4e}',flush=True)


