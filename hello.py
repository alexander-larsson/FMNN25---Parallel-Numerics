from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print "hello world from process ", rank
if rank == 0:
 comm.send(2,dest=1,tag=1)
if rank == 1:
 print(comm.recv(source=0,tag=1))
