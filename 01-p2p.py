"""
01-p2p.py: Blocking point-to-point communication

Based on the first example "Point-to-Point Communication"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

   mpiexec -n 2 python 01-p2p.py

to run on 2 processors.

This illustrates BLOCKING point-to-point communication.

Process 0 sends {'a': 3} to Process 1, blocks until it is safe to modify
the data it sent, then modifies the value of key 'a' by doubling it.

The receiver (1) blocks until it has received the data for process 0.

Both processes report the contents of their data.
(Any processes beyond 2 report None for their data.)
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID
data = None                             # For any processors with rank > 1

if rank == 0:
    n_procs = comm.Get_size()
    print(f'Running on {n_procs} processors')
    data = {'a': 3}                     # Could be anything pickleable
    comm.send(data, dest=1, tag=0)      # Blocking send for Python object
                                        # Tag is optional, but must match
                                        # if used
    data['a'] *= 2                      # Safe to modify now
elif rank == 1:
    data = comm.recv(source=0, tag=0)   # Blocking receive for Python object
                                        # Tag is optional

print(f'P{rank}: {data}')               # Runs on all processors
