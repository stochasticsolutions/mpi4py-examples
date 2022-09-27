"""
06-gather.py: Collation of results from all processors into a list
              on a nominated processor.

Based on the sixth example "Gathering Python objects"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 06-gather.py

to run on 4 processors.

Gather operations construct a list on the specified root with
a result from each process.

"""

from mpi4py import MPI

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID
size = comm.Get_size()

part = (rank, rank ** 2)
combined = comm.gather(part, root=0)    # data is assembled at root only
                                        # Blocking gather operation

if rank == 0:
    print(f'Number of processes: {size}')
    for i in range(size):
        assert combined[i] == (i, i ** 2)   # Confirm results as expected
else:
    assert combined is None

print(f'Rank {rank}: part: {part}  combined: {combined}')
