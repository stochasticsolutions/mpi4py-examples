"""
7-broadcast-numpy-array.py

Based on the seventh example "Broadcasting a Numpy array"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 7-broadcast-numpy-array.py

"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    n_procs = comm.Get_size()
    print(f'Number of processes: {n_procs}')
    data = np.arange(20, dtype='i')
else:
    data = np.empty(20, dtype='i')
comm.Bcast(data, root=0)
assert (data == np.arange(20, dtype='i')).all()

print(f'Rank {rank}: data: {data}')
