#!/usr/bin/env python
"""
13-dynamic-process-worker.py

Based on the eighth example "Dynamic Process Management"
in the mpi4py documentation
(as of 2022-06-07)

Run by 13-dynamic-process-master.py

"""
import numpy as np

from mpi4py import MPI

VERBOSE = False
vprint = print if VERBOSE else lambda m: None

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

N = np.array(0, dtype='i')  #  Scalar zero (value doesn't matter,
                            #  will be overwritten by broadcast)


vprint(f'Worker {rank} awaiting broadcast')
comm.Bcast([N, MPI.INT], root=0)  # Sets N to number of iterations
vprint(f'Worker {rank} received broadcast')

x = (np.arange(rank, N, size) + 0.5) / N
partial_pi = (4.0 / (1 + np.power(x, 2))).sum() / N
PI = np.array(partial_pi, dtype='d')

vprint(f'Worker {rank} reducing')
comm.Reduce([PI, MPI.DOUBLE], None, op=MPI.SUM, root=0)
vprint(f'Worker {rank} reduced')

comm.Disconnect()
