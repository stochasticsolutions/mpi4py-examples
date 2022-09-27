"""
13-dynamic-process-master.py

Based on the eighth example "Dynamic Process Management"
in the mpi4py documentation
(as of 2022-06-07)

Run with:

    time python 13-dynamic-process-master.py PROCS [ITERS [p|n]

where N is the total number of processes, including the master process,
ITERS is the total number of points to evaluate in calculating PI
and p means use a Python loop over the numpy array, while n means
use numpy vector operations.
"""

import sys
import numpy

from mpi4py import MPI

if len(sys.argv) < 2:
    print('USAGE: python 13-dynamic-process-master.py '
          'TOTAL-PROCS [ITERS [N|P]', file=sys.stderr)
    sys.exit(1)

iterations = 100_000_000
suffix = 'p'  # Python
if len(sys.argv) >= 3:
    iterations = int(sys.argv[2])
    if len(sys.argv) >= 4:
        suffix = sys.argv[3].lower()
        assert suffix in 'np'

n_procs = int(sys.argv[1])
print(f'Manager spawning {n_procs - 1} processes')

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=[f'13-dynamic-process-worker-{suffix}.py'],
                           maxprocs=n_procs - 1)
print(f'Manager spawned {n_procs - 1} processes')

N = numpy.array(iterations, 'i')  # This is a scalar (0-dimensional) array
                                  # containing a single number, iterations

print('Manager broadcasting')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
print('Manager sent broadcast')
PI = numpy.array(0.0, 'd')
print('Manager reducing')
comm.Reduce(None, [PI, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
print('Manager reduced')
print(PI.item())

comm.Disconnect()
