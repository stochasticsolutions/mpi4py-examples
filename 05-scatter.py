"""
05-scatter.py: Scattering of one item of iterable data of right size,
               from a nominated process to every process.

Based on the fifth example "Scattering Python objects"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 05-scatter.py

to run on 4 processors.

Scatter operations send one element from a list (or other iterable)
to each process, from a specified root process.
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID
size = comm.Get_size()

if rank == 0:
    n_procs = comm.Get_size()
    print(f'Number of processes: {n_procs}')
    data = [(i, i ** 2) for i in range(size)]  # Data must be same size as
                                               # number of processes
                                               # Can be any iterable
else:
    data = None

part = comm.scatter(data, root=0)              # Process i gets data[i]
                                               # Blocking operation
assert part == (rank, rank ** 2)


# Note that part is set for all processes, but data remains set
# only for proc 0

print(f'Rank {rank}:   part: {part}   data: {data}')
