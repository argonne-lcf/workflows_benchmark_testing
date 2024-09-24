import numpy as np
import math
from mpipartition import Partition
from torch_geometric.nn import knn_graph

PI = math.pi

# Setup problem
def setup_problem(args, comm):
    """Setup the simulation problem
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    problem_def = {'n_nodes': 1,
                   'n_features': 1,
                   'n_targets': 1,
                   'spatial_dim': 1,
                   'coords': np.empty([1]),
                   'edge_index': np.empty([1]),
    }

    if (args.problem_size=="small"):
        N = 32
        problem_def['n_nodes'] = N**2
        problem_def['n_features'] = 1
        problem_def['n_targets'] = 1
        problem_def['spatial_dim'] = 2
        partition = Partition(dimensions=spatial_dim, comm=comm)
        part_origin = partition.origin
        part_extent = partition.extent
        x = np.linspace(part_origin[0],part_origin[0]+part_extent[0],num=N,dtype=np.float64)*4*PI-2*PI
        y = np.linspace(part_origin[1],part_origin[1]+part_extent[1],num=N,dtype=np.float64)*4*PI-2*PI
        X, Y = np.meshgrid(x, y)
        problem_def['coords'] = np.vstack((X.flatten(),Y.flatten())).T
     
    problem_def['edge_index'] = setup_graph(coords)
    return problem_def


# Set up graph based on mesh coordinates
def setup_graph(coords: np.ndarray) -> np.ndarray:
    """Create local graph based on local mesh nodes
    """
    if coords.ndim<2:
        coords = np.expand_dims(coords, axis=1)
    return knn_graph(torch.from_numpy(coords), k=2, loop=False).numpy()


# Perform a step of the simulation
def simulation_step(problem_size: str, coords: np.ndarray):
    """Perform a step of the simulation
    """
    if (args.problem_size=='small'):
        r = np.sqrt(x**2+y**2)
        period = 60
        freq = 2*PI/period
        u = np.sin(2.0*r-freq*step)/(r+1.0)
        udt = np.sin(2.0*r-freq*(step+1))/(r+1.0)
        data = np.empty((n_samples,ndTot))
        data[:,0] = u.flatten()
        data[:,1] = udt.flatten()
    return data


