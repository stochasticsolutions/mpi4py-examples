"""
04-broadcast.py: Blocking broadcsat of Python object to all procs

Based on the fourth example "Broadcasting a Python Dictionary"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

    mpiexec -n 4 python 04-broadcast.py

to run on 4 processors.

This sends a dictionary to all processors using a blocking broadcast
(bcast) operation.
"""

import numpy

from mpi4py import MPI
from pprint import pformat as pf        # Pretty print formatter

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID

if rank == 0:
    n_procs = comm.Get_size()
    print(f'Number of processes: {n_procs}')

    # Data could be any pickleable data.
    # Sending numpy arrays this way is inefficient, but works.
    # (Use buffered version (Bcast) for better performance
    #  for numpy arrays.)
    data = {
        'Python list': [7, 2.72, 2+3j],
        'Python dict': {'a': 1},
        'Numpy int array': numpy.arange(5, dtype=numpy.int64),
        'Numpy reverse float array': numpy.arange(5, 0, -1, dtype=numpy.float64)
    }
else:
    data = None

data = comm.bcast(data, root=0)  # All processors execute (blocking) broadcast
                                 # Root sends; all receive

print(f'Rank {rank}: data:\n{pf(data, indent=4)}')
