#!/bin/sh

echo "mpiexec -n 4 python 1-p2p.py:"
mpiexec -n 4 python 1-p2p.py
echo
echo

echo "mpiexec -n 4 python 2-non-blocking.py:"
mpiexec -n 4 python 2-non-blocking.py
echo
echo

echo "mpiexec -n 4 python 3-numpy-arrays.py:"
mpiexec -n 4 python 3-numpy-arrays.py
echo
echo

echo "mpiexec -n 4 python 4-broadcast.py:"
mpiexec -n 4 python 4-broadcast.py
echo
echo

echo "mpiexec -n 4 python 5-scatter.py"
mpiexec -n 4 python 5-scatter.py
echo
echo

echo "mpiexec -n 4 python 6-gather.py"
mpiexec -n 4 python 6-gather.py
echo
echo

echo "mpiexec -n 4 python 7-broadcast-numpy-array.py"
mpiexec -n 4 python 7-broadcast-numpy-array.py
echo
echo


echo "mpiexec -n 4 python 8-scatter-numpy-arrays.py"
mpiexec -n 4 python 8-scatter-numpy-arrays.py
echo
echo


echo "mpiexec -n 4 python 9-gather-numpy-arrays.py"
mpiexec -n 4 python 9-gather-numpy-arrays.py
echo
echo




