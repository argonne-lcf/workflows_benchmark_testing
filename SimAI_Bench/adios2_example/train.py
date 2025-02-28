from mpi4py import MPI
import numpy as np
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

# Read setup data
with Stream(io, "setup", "r", comm) as stream:
    stream.begin_step()
    nproc_sim = stream.read("nproc")
    print(f'Trainer [{rank}]: simulation running with {nproc_sim} processes',flush=True)
    stream.end_step()

# Read training data
with Stream(io, "train_data", "r", comm) as stream:
    stream.begin_step()    
    var = stream.inquire_variable("y")
    shape = var.shape()
    count = int(shape[0] / size)
    start = count * rank
    if rank == size - 1:
        count += shape[0] % size
    train_data = stream.read("y", [start], [count])
    currentStep = stream.current_step()
    loopStep = stream.loop_index()
    stream.end_step()

# Imitating training steps
print(f"Trainer [{rank}]: training with data = {train_data}",flush=True)
sleep(3.0)

# Send trained model back
with Stream(io, "model", "w", comm) as stream:
    stream.begin_step()
    if rank == 0:
        stream.write("model", 43)
        print(f'Trainer [{rank}]: send model',flush=True) 
    stream.end_step()




