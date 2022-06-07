"""
8-scatter-numpy-arrays.py

Based on the eighth example "Scattering Numpy Arrays"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 8-scatter-numpy-arrays.py

"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = None
if rank == 0:
    n_procs = comm.Get_size()
    print(f'Number of processes: {size}')
    sendbuf = np.empty([size, 10], dtype='i')
    sendbuf.T[:,:] = range(size)
    print(f'Rank {rank}: sendbuf:\n{sendbuf}\n')
recvbuf = np.empty(10, dtype='i')
comm.Scatter(sendbuf, recvbuf, root=0)
assert np.allclose(recvbuf, rank)
print(f'Rank {rank}: recvbuf: {recvbuf}')


