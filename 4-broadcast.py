"""
4-broadcast.py

Based on the fourth example "Broadcasting a Python Dictionary"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 4-broadcast.py

"""


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    n_procs = comm.Get_size()
    print(f'Number of processes: {n_procs}')
    data = {'key1' : [7, 2.72, 2+3j],
            'key2' : ( 'abc', 'xyz')}
else:
    data = None
data = comm.bcast(data, root=0)

print(f'Rank {rank}: data: {data}')
