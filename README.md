# FMNN25---Parallel-Numerics

## How to switch virtual environments in Anaconda

http://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/

Fast way to switch to Python 2 if you are on Python 3:

conda create -n python2 python=2.7 anaconda
(this command installs it, you only have to do this once)

source activate python2
(this activates your installed env)

How to go back to default python version:

source deactivate

List your installed environments:

conda info -e

## How to install mpi4py if you have Anaconda
-mpi4py only works with Python 2

conda install --channel mpi4py mpich mpi4py

## Link to MPI tutorial

http://materials.jeremybejarano.com/MPIwithPython/

## How to run python script with MPI

Example: mpiexec -n 5 python file.py

Here the number after "-n" is the number of parallel files to run and "file.py"
is the file.
