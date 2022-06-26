"""
02-non-blocking.py: Non-blocking point-to-point communication

Based on the second example "Non-blocking Communication"
in the mpi4py documentation
(as of 2022-03-01, commit d4ae0e73493ba319a3794db6644201b9a8a548e3).

Run with:

   mpiexec -n 4 python 02-non-blocking.py

to run on 2 processors.

This illustrates NON-BLOCKING point-to-point communication.

Process 0 sends {'a': 3} to Process 1, without blocking.
If it modified the data now, this might affect what Process 1 receives.
It then waits on the request and only then modifies the value of key 'a'
by doubling it.

Process 1 sets up a non-blocking receive, then does something else
(in this case sleep, but it could be useful work) then waits
on the request, which returns the data.

Both processes report the contents of their data.
(Any processes beyond 2 report None for their data.)
"""

import time
from mpi4py import MPI

comm = MPI.COMM_WORLD                   # The MPI Intercom
rank = comm.Get_rank()                  # Processor ID
data = None                             # For any processors with rank > 1

if rank == 0:
    n_procs = comm.Get_size()
    print(f'Running on {n_procs} processors.')
    data = {'a': 3}
    req = comm.isend(data, dest=1)   # Blocking send returns request
                                     # Didn't use optional tag this time
    req.wait()                       # Wait on request (avoids leaks)
    data['a'] *= 2                   # Modify once safe to do so
elif rank == 1:
    req = comm.irecv(source=0)       # Blocking receive returns request
                                     # Again, no tag this time
    time.sleep(1)                    # Could do other useful work here;
                                     # in this case, just sleep for a second.
    data = req.wait()                # Wait on request to get data

print(f'P{rank}: data: {data}')
