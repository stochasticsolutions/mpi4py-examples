"""
3-numpy-arrays.py

ased on the third example "Numpy arrays (the fast way!)":
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

   mpiexec -n 4 python 3-numpy-arrays.py

"""


from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = None

# passing MPI datatypes explicitly
if rank == 0:
    n_procs = comm.Get_size()
    print(f'Number of processes: {n_procs}')
    data = numpy.arange(10, dtype='i')
    comm.Send([data, MPI.INT], dest=1, tag=77)
elif rank == 1:
    data = numpy.empty(10, dtype='i')
    comm.Recv([data, MPI.INT], source=0, tag=77)
print(f'Rank {rank}: data: {data}')


# automatic MPI datatype discovery
if rank == 0:
    data = numpy.arange(1, 6, dtype=numpy.float64)
    data = numpy.round(numpy.power(data, -1), 2)
    comm.Send(data, dest=1, tag=13)
elif rank == 1:
    data = numpy.empty(5, dtype=numpy.float64)
    comm.Recv(data, source=0, tag=13)
print(f'Rank {rank}: data: {data}')
