from mpi4py import MPI
import numpy as np
from time import sleep, perf_counter
from adios2 import Stream, Adios, bindings

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

# note: we need to use np.float32 to be compatible with data from C++ writer
# using "float" works in Python only but leads to type mismatch with C++
N = 10_000_000 
myArray = np.arange(N, dtype=np.float64)
myArray = 10.0 * rank + myArray
nx = len(myArray)

# Send some data
tic = perf_counter()
with Stream(io, "data", "w", comm) as stream:
    stream.begin_step()
    ticc = perf_counter()
    stream.write("y", myArray, [size * nx], [rank * nx], [nx])
    print(f'[S{rank}] write time: {perf_counter()-ticc:>4e}',flush=True)
    sleep(5)
    ticc = perf_counter()
    stream.end_step()
    print(f'[S{rank}] end_step time: {perf_counter()-ticc:>4e}',flush=True)
print(f'[S{rank}] total send time: {perf_counter()-tic:>4e}',flush=True)

# Sleep for a little otherwise simulation model reader throws exception that it's writer closed too soon
sleep(1)

