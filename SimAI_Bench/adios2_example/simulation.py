from mpi4py import MPI
import numpy as np
from time import sleep
from adios2 import Stream, Adios, bindings

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ADIOS MPI Communicator
adios = Adios(comm)

# ADIOS IO
io = adios.declare_io("myIO")
io.set_engine("Sst")
parameters = {
    'RendezvousReaderCount': '1', # options: 1 for sync, 0 for async
    'QueueFullPolicy': 'Block', # options: Block, Discard
    'DataTransport': 'WAN' # options: MPI, WAN,  UCX, fabric
}
io.set_parameters(parameters)

# note: we need to use np.float32 to be compatible with data from C++ writer
# using "float" works in Python only but leads to type mismatch with C++
myArray = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
myArray = 10.0 * rank + myArray
nx = len(myArray)
increment = nx * size * 1.0

# Send some setup data
with Stream(io, "setup", "w", comm) as stream:
    stream.begin_step()
    if rank == 0:
        stream.write("nproc", size)
        print(f'Simulation [{rank}]: sent setup data',flush=True) 
    stream.end_step()

# Imitating simulation steps
sleep(2.0)

# Send training data
with Stream(io, "train_data", "w", comm) as stream:
    stream.begin_step()
    stream.write("y", myArray, [size * nx], [rank * nx], [nx])
    myArray += increment
    print(f"Simulaiton [{rank}]: sent data = {myArray}", flush=True)
    stream.end_step()

# Read model
with Stream(io, "model", "r", comm) as stream:
    stream.begin_step()
    model = stream.read("model")
    stream.end_step()
    print(f'Simulation [{rank}]: simulation read model {model}',flush=True) 



