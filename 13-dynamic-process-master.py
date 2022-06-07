"""
13-dynamic-process-master.py

Based on the eighth example "Dynamic Process Management"
in the mpi4py documentation
(as of 2022-06-07)

Run with:

    python 13-dynamic-process-master.py  N

where N is the number of processes, including the master process.

"""
from mpi4py import MPI
import numpy
import sys

if len(sys.argv) < 2:
    print('USAGE: python 13-dynamic-process-master.py  N-PROCS-TOTAL', file=sys.stderr)
    sys.exit(1)
n_procs = int(sys.argv[1])
comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['13-dynamic-process-worker.py'],
                           maxprocs=n_procs - 1)

N = numpy.array(100000000, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
PI = numpy.array(0.0, 'd')
comm.Reduce(None, [PI, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
print(PI)

comm.Disconnect()
