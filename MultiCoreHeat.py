#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from mpi4py import MPI
import numpy as np
#import matplotlib.pyplot as plt
# Lägg till så man kan säga vilken sida Neumann vilkoret ligger på (höger/vänster)

def make_A_matrix_small_room(size,side):
    """
    size: size of room matrix (scalar)
    side: the side where the Neuman condition is (L/R)
    """
    A_size = (size-1)*(size-2)
    A = np.diag(np.ones(A_size)*-4)
    for i in range(1,A_size):
        if i%(size-1):
            A[i-1][i] = 1
            A[i][i-1] = 1
    for i in range(size-1,A_size):
        A[i-size+1][i] = 1
        A[i][i-size+1] = 1

    # Add twos for Neumann condition

    for i in range(size-2):
        if side == "L":
            bi = (size - 1)*i
            A[bi][bi+1] = 2
        elif side == "R":
            bi = (size - 2) + (size-1)*i
            A[bi][bi-1] = 2

    return A

def make_A_matrix_big_room(size):
    """
    Parameters:
    size: tuple on the form (m,n)
    """
    # indexes using m and n is probably wrong
    m,n = size
    A_size = (m-2)*(n-2)
    A = np.diag(np.ones(A_size)*-4)
    for i in range(1,A_size):
        if i%(n-2):
            A[i-1][i] = 1
            A[i][i-1] = 1
    for i in range(A_size-n+2):
        A[i][i+n-2] = 1
        A[i+n-2][i] = 1
    return A


def make_B_vector(left,top,right,bottom):
    """
    Parameters:
    4 vectors corresponding to the elements on either side of the matrix.
    If the matrix is of size n x m the top and bottom vectors will be n-2.
    The left and right will be m-2.
    For left and right 0 is the top element.
    """
    if not left.shape == right.shape or not top.shape == bottom.shape:
        # Raise exception instead
        print("Error, wrong shapes")
    rows = left.size
    cols = top.size

    length = rows*cols

    b = np.zeros(length)

    # Hörn 1
    b[0] = top[0]+left[0]

    # rows - 2 element top
    b[1:cols-1] = top[1:-1]

    # Hörn 2
    b[cols-1] = top[-1]+right[0]

    # Mid points
    for i in range(1,rows-1):
        b[i*cols] = left[i]
        b[i*cols+cols-1] = right[i]

    # Hörn 3
    b[length-cols] = bottom[0]+left[-1]

    # rows - 2 element bottom
    b[length-cols+1:-1] = bottom[1:-1]

    # Hörn 4
    b[length-1] = bottom[-1]+right[-1]

    return b

def update_B_vector_big_room(b, interface_points_left,interface_points_right):

    ##  Insert the calculated points in b
    cols = interface_points_left.size
    u = 0
    k = 0
    print(b.size)
    print(cols)
    rows = b.size // interface_points_left.size
    print(rows)
    for i in range(1,rows-1):
        if i < (rows)/2: #Remember to check index
            b[i*cols+cols-1] = 1 #interface_points_right[u]
            u = u + 1
        else:
            b[i*cols] = 1 #interface_points_left[k]
            k = k + 1
    return b

def update_B_vector_small_room(b,interface_side,guess):
    """
    Parameters:
    b: the vector created by running _make_B_matrix_
    neumann_side: the string "L" or "R" to specify where the interface is
    """
def gamma(u_l, u_r):
    return (u_l - u_r)*inv_dx/2
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


inv_dx = 3
#Intialize rooms

if rank == 0:
    left = np.ones(inv_dx*2-1)*15
    top = np.ones(inv_dx-1)*40
    bottom = np.ones(inv_dx-1)*5
    right = np.ones(inv_dx*2-1)*15
    a = make_A_matrix_big_room((inv_dx*2 + 1,inv_dx + 1))
    b = make_B_vector(left,top,right,bottom)
    x_middle = np.linalg.solve(a,-b)
    comm.Bcast(x_middle, root=0)
if rank != 0:
    top = np.ones(inv_dx)*15
    bottom = np.ones(inv_dx)*15
    right = np.ones(inv_dx-1)*40    
    left = np.ones(inv_dx-1)*40
    if rank == 1:
        for i in range(right.size):
            right = x_middle[-right.size*(i+1)]
        right = right[::-1] #Reverse array
        a = make_A_matrix_small_room(inv_dx +1, "L")
        b = make_B_vector(left,top,right,bottom)
        x = np.linalg.solve(a, -b)
        #interface_points_left = x_left.reshape(inv_dx-1,inv_dx)[:,inv_dx-1]
        #Tag with zero for left
        #SEND: 
        x_old = x.reshape(inv_dx-1, inv_dx)[:,1]
        comm.Send(x.reshape(inv_dx-1,inv_dx)[:,inv_dx-1], dest=0, tag=1)
    elif rank == 2:
        for i in range(left.size):
            left[i] = x_middle[left.size*(i)]
        a = make_A_matrix_small_room(inv_dx +1, "R")
        b = make_B_vector(left,top,right,bottom)
        x = np.linalg.solve(a, -b)
        x_old = x.reshape(inv_dx-1,inv_dx)[:,1]
        #Send the interface points
        interface_points_left = x.reshape(inv_dx-1, inv_dx)[:,0]
        print(interface_points_left)
        comm.Send(interface_points_left, dest=0, tag=2)
elif rank > 2:
    print("Rank ", rank, " does nothing")
    
#Done, iterate
comm.Barrier()
for _ in range(1,10):
    #First the big room
    if rank == 0:
        interface_points_left = np.ones([1,2])
        comm.Recv(interface_points_left, source = 1, tag=1)
        comm.Recv(interface_points_right, source = 2, tag=2)
        right[:interface_points_left.size] = interface_points_left
        left[-interface_points_right.size:] = interface_points_right
        b = make_B_vector(left,top,right,bottom)
        x_middle = np.linalg.solve(a,-b)
        comm.Bcast(x_middle, root=0)
    comm.Barrier()
    if rank != 0:
        #Common stuff
        gamma_v = np.zeros(inv_dx-1)
        temp = x_middle.reshape(inv_dx*2 -1,inv_dx-1)
        for i in range(gamma.size):
            gamma[i] = gamma(x_old[i], temp[-(inv_dx-1):,0][i])
        if rank == 1: #left
            b = make_B_vector(left,top,((2/inv_dx)*gamma_v),bottom)
        elif rank == 2: #right
            b = make_B_vector(((2/inv_dx)*gamma_v),top, right ,bottom)
            b = make_B_vector(((2/inv_dx)*gamma_v),top, right, bottom)
        x = np.linalg.solve(a, -b)
        x.old = x.reshape(inv_dx-1,inv_dx)[:,1]
        if rank == 1:
            comm.send(x.reshape(inv_dx-1,inv_dx)[:,inv_dx-1], dest=0, tag=1)
        elif rank == 2:
            comm.send(x.reshape(inv_dx-1,inv_dx)[:,0])
            #Broadcast points        
if rank == 0:
    print(x)