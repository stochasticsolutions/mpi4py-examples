"""
6-gather.py

Based on the sixth example "Gathering Python objects"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 6-gather.py

"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = (rank+1)**2
data = comm.gather(data, root=0)
if rank == 0:
    print(f'Number of processes: {size}')
    for i in range(size):
        assert data[i] == (i+1)**2
else:
    assert data is None

print(f'Rank {rank}: data: {data}')

