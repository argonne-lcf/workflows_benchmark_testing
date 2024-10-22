from time import sleep
import numpy as np
import math
from mpipartition import Partition

import torch
from torch_geometric.nn import knn_graph

PI = math.pi

# Setup problem
def setup_problem(args, comm):
    """Setup the simulation problem
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    problem_def = {'n_nodes': 1,
                   'n_edges': 1,
                   'n_features': 1,
                   'n_targets': 1,
                   'spatial_dim': 1,
                   'coords': np.empty([1]),
                   'edge_index': np.empty([1]),
    }

    if args.problem_size=="small":
        N = 32 #2_000 // size
        problem_def['n_nodes'] = N**2
        problem_def['n_features'] = 1
        problem_def['n_targets'] = 1
        problem_def['spatial_dim'] = 2
        problem_def['coords'] = np.empty((problem_def['n_nodes'], problem_def['spatial_dim']), dtype=np.double)
        partition = Partition(dimensions=problem_def['spatial_dim'], comm=comm)
        part_origin = partition.origin
        part_extent = partition.extent
        x = np.linspace(part_origin[0],part_origin[0]+part_extent[0],num=N,dtype=np.double)*4*PI-2*PI
        y = np.linspace(part_origin[1],part_origin[1]+part_extent[1],num=N,dtype=np.double)*4*PI-2*PI
        X, Y = np.meshgrid(x, y)
        problem_def['coords'][:,0] = X.flatten()
        problem_def['coords'][:,1] = Y.flatten()
    if args.problem_size=="medium":
        N = 256 #1_000_000 // size
        problem_def['n_nodes'] = N**2
        problem_def['n_features'] = 2
        problem_def['n_targets'] = 2
        problem_def['spatial_dim'] = 2
        problem_def['coords'] = np.empty((problem_def['n_nodes'], problem_def['spatial_dim']), dtype=np.double)
        partition = Partition(dimensions=problem_def['spatial_dim'], comm=comm)
        part_origin = partition.origin
        part_extent = partition.extent
        x = np.linspace(part_origin[0],part_origin[0]+part_extent[0],num=N,dtype=np.double)*4*PI-2*PI
        y = np.linspace(part_origin[1],part_origin[1]+part_extent[1],num=N,dtype=np.double)*4*PI-2*PI
        X, Y = np.meshgrid(x, y)
        problem_def['coords'][:,0] = X.flatten()
        problem_def['coords'][:,1] = Y.flatten()
    if args.problem_size=="large":
        N = 100 #100_000_000 // size
        problem_def['n_nodes'] = N**3
        problem_def['n_features'] = 3
        problem_def['n_targets'] = 3
        problem_def['spatial_dim'] = 3
        problem_def['coords'] = np.empty((problem_def['n_nodes'], problem_def['spatial_dim']), dtype=np.double)
        partition = Partition(dimensions=problem_def['spatial_dim'], comm=comm)
        part_origin = partition.origin
        part_extent = partition.extent
        x = np.linspace(part_origin[0],part_origin[0]+part_extent[0],num=N,dtype=np.double)*4*PI-2*PI
        y = np.linspace(part_origin[1],part_origin[1]+part_extent[1],num=N,dtype=np.double)*4*PI-2*PI
        z = np.linspace(part_origin[2],part_origin[2]+part_extent[2],num=N,dtype=np.double)*4*PI-2*PI
        X, Y, Z = np.meshgrid(x, y, z)
        problem_def['coords'][:,0] = X.flatten()
        problem_def['coords'][:,1] = Y.flatten()
        problem_def['coords'][:,2] = Z.flatten()
     
    problem_def['edge_index'] = setup_graph(problem_def['coords'])
    problem_def['n_edges'] = problem_def['edge_index'].shape[1]
    return problem_def


# Set up graph based on mesh coordinates
def setup_graph(coords: np.ndarray) -> np.ndarray:
    """Create local graph based on local mesh nodes
    """
    if coords.ndim<2:
        coords = np.expand_dims(coords, axis=1)
    return knn_graph(torch.from_numpy(coords), k=2, loop=False).numpy().astype('int64')


# Perform a step of the simulation
def simulation_step(step: int, problem_size: str, coords: np.ndarray):
    """Perform a step of the simulation
    """
    n_samples = coords.shape[0]
    if problem_size=='small':
        r = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
        period = 60
        freq = 2*PI/period
        u = np.sin(2.0*r-freq*step)/(r+1.0)
        udt = np.sin(2.0*r-freq*(step+1))/(r+1.0)
        data = np.empty((n_samples,2))
        data[:,0] = u.flatten()
        data[:,1] = udt.flatten()
        sleep(0.1)
    if problem_size=='medium':
        r = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
        period = 100
        freq = 2*PI/period
        u = np.sin(2.0*r-freq*step)/(r+1.0)
        udt = np.sin(2.0*r-freq*(step+1))/(r+1.0)
        v = np.cos(2.0*r-freq*step)/(r+1.0)
        vdt = np.cos(2.0*r-freq*(step+1))/(r+1.0)
        data = np.empty((n_samples,4))
        data[:,0] = u.flatten()
        data[:,1] = v.flatten()
        data[:,2] = udt.flatten()
        data[:,3] = vdt.flatten()
        sleep(0.3)
    if problem_size=='large':
        r = np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2)
        period = 200
        freq = 2*PI/period
        u = np.sin(2.0*r-freq*step)/(r+1.0)
        udt = np.sin(2.0*r-freq*(step+1))/(r+1.0)
        v = np.cos(2.0*r-freq*step)/(r+1.0)
        vdt = np.cos(2.0*r-freq*(step+1))/(r+1.0)
        w = np.sin(2.0*r-freq*step)**2/(r+1.0)
        wdt = np.sin(2.0*r-freq*(step+1))**2/(r+1.0)
        data = np.empty((n_samples,6))
        data[:,0] = u.flatten()
        data[:,1] = v.flatten()
        data[:,2] = w.flatten()
        data[:,3] = udt.flatten()
        data[:,4] = vdt.flatten()
        data[:,5] = wdt.flatten()
        sleep(0.5)

    return data


# GMRES
def GMRES(A, b, x0=None, P=None, tol=1e-5, max_iter=200, restart=None):
    """Solve the linear system Ax=b via Generalized Minimal RESidual
    
    Implemented in PyTorch

    Reference:
        M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
        Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).

    .. seealso:: https://github.com/cupy/cupy/blob/v13.3.0/cupyx/scipy/sparse/linalg/_iterative.py
    """
    n = A.shape[0]
    dtype = A.dtype
    np_dtype = torch.zeros((1,),dtype=dtype).numpy().dtype
    device = A.device
    
    if n == 0:
        return torch.zeros_like(b)
    
    b_norm = torch.linalg.norm(b)
    if b_norm == 0:
        return b
    
    if restart is None:
        restart = max_iter
    restart = min(restart, min(n, max_iter))

    A, P, x, b = make_system(A, P, x0, b)

    Q = torch.empty((n, restart), dtype=dtype, device=device)
    H = torch.zeros((restart+1, restart), dtype=dtype, device=device)
    e = numpy.zeros((restart+1,), dtype=np_dtype)

    compute_hu = _make_compute_hu(V)

    iters = 0
    while True:
        r = b - torch.matmul(A,x)
        r_norm = torch.linalg.norm(r)
        if r_norm <= tol or iters >= max_iter:
            break
        r = r / r_norm
        Q[:, 0] = r
        e[0] = r_norm

        # Arnoldi iteration
        for j in range(restart):
            Q[:,j+1] = torch.matmul(A,Q[:,j])
            for i in range(j):
                H[i,j] = torch.dot(Q[:,i],Q[:,j+1])
                Q[:,j+1] = Q[:,j+1] - H[i,j]*Q[:,i]
            H[j+1,j] = torch.linalg.norm(Q[:,j+1])
            if torch.abs(H[j+1,j])>tol:
                Q[:,j+1] = Q[:,j+1] / H[j+1,j]
            


u = matvec(z)
            H[:j+1, j], u = compute_hu(u, j)
            cublas.nrm2(u, out=H[j+1, j])
            if j+1 < restart:
                v = u / H[j+1, j]
                V[:, j+1] = v

        # Note: The least-square solution to equation Hy = e is computed on CPU
        # because it is faster if the matrix size is small.
        ret = numpy.linalg.lstsq(cupy.asnumpy(H), e)
        y = cupy.array(ret[0])
        x += V @ y
        iters += restart

    info = 0
    if iters == maxiter and not (r_norm <= atol):
        info = iters
    return mx


# Make system of equations
def make_system(A, P, x0, b):
    """Make linear system of equations
    """
    n = A.shape[0]
    dtype = A.dtype
    device = A.device
    if x0 is None:
        x = torch.zeros((n,), dtype=dtype, device=device)
    else:
        if not (x0.shape == (n,) or x0.shape == (n, 1)):
            raise ValueError('x0 has incompatible dimensions')
        x = x0
        if x.dtype != dtype: x = x.type(dtype)
        if x.device != device: x.to(device)
    if P is None:
        P = torch.eye(n, dtype=dtype, device=device)
    else:
        if A.shape != P.shape:
            raise ValueError('matrix and preconditioner have different shapes')
        if P.dtype != dtype: P = P.type(dtype)
        if P.device != device: P.to(device)
    return A, P, x, b



