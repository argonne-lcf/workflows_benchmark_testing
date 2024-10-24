import sys
from time import sleep, perf_counter
import numpy as np
import math
from mpipartition import Partition

import torch
from torch_geometric.nn import knn_graph

from scipy import sparse
from scipy.sparse import linalg as spla

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
def gmres(A, b, x0=None, P=None, tol=1e-5, max_iter=200, restart=None, logging=False):
    """Solve the linear system Ax=b via Generalized Minimal RESidual
    
    Implemented in PyTorch

    #Reference:
    #    M. Wang, H. Klie, M. Parashar and H. Sudan, "Solving Sparse Linear
    #    Systems on NVIDIA Tesla GPUs", ICCS 2009 (2009).
    #
    #.. seealso:: https://github.com/cupy/cupy/blob/v13.3.0/cupyx/scipy/sparse/linalg/_iterative.py
    """
    n = A.shape[0]
    dtype = A.dtype
    np_dtype = torch.zeros((1,),dtype=dtype).numpy().dtype
    device = A.device
    if logging: print(f'Executing GMRES with precision {dtype} and on device {device}',flush=True)   
 
    if n == 0:
        return torch.zeros_like(b), 0, 0
    
    b_norm = torch.linalg.norm(b)
    if b_norm == 0:
        return b, b_borm, 0
    
    if restart is None:
        restart = max_iter
    n_krylov = min(n, min(restart, max_iter))
    assert n_krylov>=2, 'Number of max_iter or restart must be >= 2'
    if logging: print(f'Using {n_krylov} Krylov vectors',flush=True)   

    A, P, x, b = make_system(A, P, x0, b)

    # consider switching rows and columns of Q and H for performance
    Q = torch.empty((n, n_krylov+1), dtype=dtype, device=device)
    H = torch.zeros((n_krylov+1, n_krylov), dtype=dtype, device=device)
    e = torch.zeros((n_krylov+1,), dtype=dtype, device=device)

    # Following cupy implementation
    # https://github.com/cupy/cupy/blob/118ade4a146d1cc68519f7f661f2c145f0b942c9/cupyx/scipy/sparse/linalg/_iterative.py#L92
    iters = 0
    while True:
        px = torch.matmul(P,x)
        res = b - torch.matmul(A,px)
        res_norm = torch.linalg.norm(res)
        if res_norm <= tol or iters >= max_iter:
            break
    
        q = res / res_norm
        Q[:, 0] = q
        e[0] = res_norm
        for j in range(n_krylov):
            z = torch.matmul(P,q)
            u = torch.matmul(A, z)
            h = torch.matmul(Q[:,:j+1].t(),u)
            u -= torch.matmul(Q[:,:j+1],h)
            H[:j+1,j] = h
            H[j+1,j] = torch.linalg.norm(u)
            q = u / H[j+1,j]
            Q[:,j+1] = q
            iters+=1
            if logging:
                y = torch.linalg.lstsq(H, e).solution
                res_norm = torch.linalg.norm(torch.matmul(H,y) - e)
                print(f'iter {iters}\tres = {res_norm.item()}',flush=True)
        
        y = torch.linalg.lstsq(H, e).solution
        res_norm = torch.linalg.norm(torch.matmul(H,y) - e)
        x += torch.matmul(Q[:,:j+1],y)


    # From https://acme.byu.edu/00000179-aa18-d402-af7f-abf806ac0001/gmres2020-pdf 17.1
    """
    res0 = b - torch.matmul(A,x0)
    res0_norm = torch.linalg.norm(res0)
    if res0_norm <= tol:
        return x0, res0_norm, 0
    
    Q[:, 0] = res0 / res0_norm
    e[0] = res0_norm
    iters = 0
    for j in range(n_krylov):
        Q[:,j+1] = torch.matmul(A, Q[:,j])
        for i in range(j):
            H[i,j] = torch.dot(Q[:,i],Q[:,j+1])
            Q[:,j+1] = Q[:,j+1] - H[i,j]*Q[:,i]
        H[j+1,j] = torch.linalg.norm(Q[:,j+1])
        if torch.abs(H[j+1,j])>tol:
            Q[:,j+1] = Q[:,j+1] / H[j+1,j]
        y = torch.linalg.lstsq(H[:j+2,:j+1], e[:j+2]).solution
        res = torch.linalg.norm(torch.matmul(H[:j+2,:j+1],y) - e[:j+2])
        print(res.item())
        x = torch.matmul(Q[:,:j+1], y) + x0
        #y = torch.linalg.lstsq(H, res0_norm*e).solution
        #res = torch.linalg.norm(torch.matmul(H,y) - res0_norm*e)
        iters+=1
        if res <= tol:
            break
    """
    return x, res_norm, iters


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
        if not x0.shape == (n,):
            raise ValueError('x0 has incompatible dimensions, must be 1D vector of length n')
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


# Callback for scipy GMRES
class scipy_gmres_callback(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print(f'iter {self.niter}\tres = {rk}')


# Check GMRES implementation
def check_gmres(N=10, device='cpu'):
    """Compare GMRES implementation to known solution or to scipy solution
    """
    torch_device = torch.device(device)
    A = np.random.rand(N, N).astype(np.float64)
    b = np.random.rand(N).astype(np.float64)
    
    # Native
    rtime = perf_counter()
    x, res, iters = gmres(
                          torch.from_numpy(A).to(torch_device),
                          torch.from_numpy(b).to(torch_device),
                          tol=1e-6,
                          logging=True
    )
    rtime = perf_counter() - rtime
    print(f'Native GMRES completed in {rtime} sec with {iters} iters and residual {res.item()}')
    
    # Scipy
    callback = scipy_gmres_callback()
    rtime = perf_counter()
    x_sp, info = spla.gmres(A, b, restart=100, atol=1e-6, callback=callback)
    rtime = perf_counter() - rtime
    if info == 0:
        print(f'Scipy GMRES converged successfully in {rtime} sec')
    else:
        print(f'Scipy GMRES completed in {rtime} sec with {info} iters')
    
    all_close = np.allclose(x.cpu().numpy(), x_sp)
    if all_close:
        print('Success')
    else:
        print('Native and scipy GMRES differ')


if __name__=='__main__':
    if len(sys.argv) == 3:
        N = int(sys.argv[1])
        device = sys.argv[2]
        check_gmres(N, device)
    else:
        check_gmres()

