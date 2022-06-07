"""
9-gather-numpy-arrays.py

Based on the eighth example "Gathering Numpy Arrays"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 9-gather-numpy-arrays.py

"""
import time
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.zeros(10, dtype='i') + rank
print(f'Rank {rank}: sendbuf:\n{sendbuf}')
recvbuf = None
time.sleep(1)
if rank == 0:
    n_procs = comm.Get_size()
    print(f'\nNumber of processes: {n_procs}\n')
    recvbuf = np.empty([size, 10], dtype='i')
comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    for i in range(size):
        assert np.allclose(recvbuf[i,:], i)
    print(f'Rank {rank}: recvbuf (gathered results):\n{recvbuf}')
else:
    print(f'Rank {rank}: recvbuf: {recvbuf}')
