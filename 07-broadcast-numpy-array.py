"""
07-broadcast-numpy-array.py: Send a numpy array to all processors,
                             using a blocking buffer broadcast (Bcast).

Based on the seventh example "Broadcasting a Numpy array"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 07-broadcast-numpy-array.py

to run on 4 processors.

This sends a numpy array to all processors using a blocking buffer broadcast
(Bcast) operation, and verifies the results.

This is similar to 04-broadcast.py, but for numpy arrays.
"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID
N = 10

if rank == 0:
    n_procs = comm.Get_size()
    print(f'Number of processes: {n_procs}')
    data = np.arange(N, dtype='i')     # data to be broadcast
else:
    data = np.empty(N, dtype='i')      # empty array (buffer) on other proces

comm.Bcast(data, root=0)                # Blocking buffer broadcast

assert (data == np.arange(N, dtype='i')).all()  # On each process, confirm
                                                 # all data items are as
                                                 # expected

print(f'Rank {rank}: data: {data}')
