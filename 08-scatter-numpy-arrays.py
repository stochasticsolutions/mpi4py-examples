"""
08-scatter-numpy-arrays.py: Scattering of one slice of a suitable
                            numpy array from a nominated process to
                            every process.

Based on the eighth example "Scattering Numpy Arrays"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 08-scatter-numpy-arrays.py

to run on 4 processors.

Scatter operations split an array whose zero dimension has size N (the number
of procs) on the 0th dimension, giving a slice to each process.
"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID

sendbuf = None                          # Buffer not needed except on proc 0
if rank == 0:
    size = comm.Get_size()
    print(f'Number of processes: {size}')
    # Create numbers 0 to 10 * (size - 1), and shape as
    # [[0, 1, ..., 9]
    #  [10, 11, ..., 19]
    #  ...
    #  [(size * 10), (size * 10) + 1, ..., size * 10 - 1]]
    sendbuf = np.arange(size * 10, dtype='i').reshape(size, 10)
    print(f'Rank {rank}: sendbuf:\n{sendbuf}\n')

recvbuf = np.empty(10, dtype='i')         # Each proc creates empty receive buf
comm.Scatter(sendbuf, recvbuf, root=0)    # Blocking scatter distributes array
                                          # by row (0 dimension)

# On each process, check what was received was the right row

assert (recvbuf == np.arange(rank * 10, (rank + 1) * 10, dtype='i')).all()

print(f'Rank {rank}: recvbuf: {recvbuf}')
