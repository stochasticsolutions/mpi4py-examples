"""
03-numpy-arrays.py: Blocking communiction of numpy arrays

Based on the third example "Numpy arrays (the fast way!)":
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

   mpiexec -n 2 python 03-numpy-arrays.py

to run on 2 processors.

This illustrates how to use Send and Recv to send buffers, in this
case numpy arrays.

Process 0 creates an integer array and sends it to process 1,
specifying the numpy type.

Process 1 then creates a floating-point array and sends it to process 1,
without specifying the numpy type (though process 1 needs to know,
to allocate the correct empty array). In the second case, we overprovision
the array to show that only the sent (5) elements are modified.

If you run with -n 3 or more, other processors should report None
for data.
"""


from mpi4py import MPI
import numpy
import time

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID
data = None                             # For any processors with rank > 1

# Pass MPI datatypes explicitly

if rank == 0:
    n_procs = comm.Get_size()
    print(f'Running on {n_procs} procs: ')
    data = numpy.arange(10, dtype='i')
    comm.Send([data, MPI.INT], dest=1)    # Blocking send for buffer
elif rank == 1:
    data = numpy.empty(10, dtype='i')     # Empty buffer for receive
    comm.Recv([data, MPI.INT], source=0)  # Blocking receive for buffer
print(f'P{rank}: data: {data}')

time.sleep(1)                             # Let those finish
if rank == 0:
    print('\n')                           # Print two blank lines from 0
time.sleep(1)                             # Wait some more

# Automatic MPI datatype discovery (on sending side)

if rank == 0:
    data = numpy.arange(1, 6, dtype=numpy.float64)
    data = numpy.round(numpy.power(data, -1), 2)  # first five reciprocals
    comm.Send(data, dest=1)                       # Blocking buffer send
elif rank == 1:
    data = numpy.ones(10, dtype=numpy.float64) * -1  # Overprovisioned buffer
                                                     # filled with -1.0
    comm.Recv(data, source=0)                        # Only data sent filled in


print(f'Rank {rank}: data: {data}')
