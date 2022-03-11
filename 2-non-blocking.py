"""
2-non-blocking.py

Based on the second example "Non-blocking Communication"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

   mpiexec -n 4 python 2-non-blocking.py

"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = None
if rank == 0:
    n_procs = comm.Get_size()
    print(f'Number of processes: {n_procs}')
    data = {'a': 7, 'b': 3.14}
    req = comm.isend(data, dest=1, tag=11)
    req.wait()
elif rank == 1:
    req = comm.irecv(source=0, tag=11)
    data = req.wait()

print(f'Rank {rank}: data: {data}')
