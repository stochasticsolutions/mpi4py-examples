"""
1-p2p.py

Based on the first example "Point-to-Point Communication"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

   mpiexec -n 4 python 01-p2p.py

"""

from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
data = None
if rank == 0:
    n_procs = comm.Get_size()
    print(f'{n_procs} procs')
    data = {'a': 3}
    comm.send(data, dest=1, tag=0)
elif rank == 1:
    data = comm.recv(source=0, tag=0)

print(f'P{rank}: {data}')
