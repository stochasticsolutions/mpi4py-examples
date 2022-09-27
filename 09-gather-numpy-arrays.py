"""
09-gather-numpy-arrays.py: Construction of a results array whose
                           slices each come from a different process.

Based on the eighth example "Gathering Numpy Arrays"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 09-gather-numpy-arrays.py

to run on 4 processors.

This gather operation construct a numpy array whose zero dimension has
the same size and the number of processes, by getting the ith slice
from process i.

Also shows use of Allgather to replicate combined result on all processors.
"""
import time

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID
size = comm.Get_size()

sendbuf = np.arange(10, dtype='i') + rank * 10
print(f'Rank {rank}: sendbuf:\n{sendbuf}')
recvbuf = None
time.sleep(1)

if rank == 0:
    print(f'\nNumber of processes: {size}\n')
    recvbuf = np.empty([size, 10], dtype='i')

comm.Gather(sendbuf, recvbuf, root=0)  # Blocking collation of results
                                       # from each sendbuf into recvbuf
                                       # on root process (0)

if rank == 0:
    assert (recvbuf.reshape(size * 10) == np.arange(size * 10)).all()
    print(f'Rank {rank}: recvbuf (gathered results):\n{recvbuf}')
    time.sleep(1)
    print()
else:
    print(f'Rank {rank}: recvbuf: {recvbuf}')


# Now use Allgather to get the result everywhere

recvbuf = np.empty([size, 10], dtype='i')  # All processors now need buffer
comm.Allgather(sendbuf, recvbuf)           # Blocking collation of results
                                           # from each sendbuf everyhwere
# Check everyone's result is right.=

assert (recvbuf.reshape(size * 10) == np.arange(size * 10)).all()
print(f'All present and correct for rank {rank}.')
