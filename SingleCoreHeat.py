#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
# Lägg till så man kan säga vilken sida Neumann vilkoret ligger på (höger/vänster)

def make_A_matrix_small_room(size,side):
    """
    size: size of room matrix (scalar)
    side: the side where the Neuman condition is (L/R)
    """
    A_size = (size-1)*(size-2)
    A = np.diag(np.ones(A_size)*-4)
    for i in xrange(1,A_size):
        if i%(size-1):
            A[i-1][i] = 1
            A[i][i-1] = 1
    for i in xrange(size-1,A_size):
        A[i-size+1][i] = 1
        A[i][i-size+1] = 1

    # Add twos for Neumann condition

    for i in xrange(size-2):
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
    for i in xrange(1,A_size):
        if i%(n-2):
            A[i-1][i] = 1
            A[i][i-1] = 1
    for i in xrange(A_size-n+2):
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
        print "Error, wrong shapes"
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
    for i in xrange(1,rows-1):
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
    for i in xrange(1,rows-1):
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

#Intialize big room
inv_dx = 3

l_middle = np.ones(inv_dx*2-1)*15
t_middle = np.ones(inv_dx-1)*40
b_middle = np.ones(inv_dx-1)*5
r_middle = np.ones(inv_dx*2-1)*15

# b_middle - Bottom row in middle room
# b_big - b-vector for middle room

a_middle = make_A_matrix_big_room((inv_dx*2 + 1,inv_dx + 1))
b_vector_middle = make_B_vector(l_middle,t_middle,r_middle,b_middle)
x_middle = np.linalg.solve(a_middle,-b_vector_middle)

#Initialize small Rooms

#Left room
l_small_left = np.ones(inv_dx-1)*40
t_small = np.ones(inv_dx)*15
b_small = np.ones(inv_dx)*15
#Insert interface points in r_small_left
#get vector with our left points from x_middle
r_small_left = np.ones(inv_dx-1)
for i in xrange(r_small_left.size):
    r_small_left[i] = x_middle[-r_small_left.size*(i+1)]
r_small_left = r_small_left[::-1] #Reverse array

a_left = make_A_matrix_small_room(inv_dx +1, "L")

b_left = make_B_vector(l_small_left,t_small,r_small_left,b_small)
x_left = np.linalg.solve(a_left, -b_left)
interface_points_left = x_left.reshape(inv_dx-1,inv_dx)[:,inv_dx-1]
#Right room
r_small_right = np.ones(inv_dx-1)*40
l_small_right = np.ones(inv_dx-1)
for i in xrange(r_small_right.size):
    l_small_right[i] = x_middle[l_small_right.size*(i)]
a_right = make_A_matrix_small_room(inv_dx +1, "R")
b_right = make_B_vector(l_small_right,t_small,r_small_right,b_small)
x_right = np.linalg.solve(a_left, -b_left)
interface_points_right = x_right.reshape(inv_dx-1,inv_dx)[:,0]
#Done, iterate
x_left_old = x_left.reshape(inv_dx-1,inv_dx)[:,1]
x_right_old = x_right.reshape(inv_dx-1, inv_dx)[:,1]
for _ in xrange(1,10):
    r_middle[:interface_points_left.size] = interface_points_left
    l_middle[-interface_points_right.size:] = interface_points_right
    b_vector_middle = make_B_vector(l_middle,t_middle,r_middle,b_middle)
    x_middle = np.linalg.solve(a_middle,-b_vector_middle)
    #start with left room
    #Update l_small_right with help from x_middle
    gamma_left = np.zeros(inv_dx-1)
    temp_x = x_middle.reshape(inv_dx*2 -1,inv_dx-1)
    for i in xrange(gamma_left.size):
        gamma_left[i] = gamma(x_left_old[i], temp_x[-(inv_dx-1):,0][i])
    b_left = make_B_vector(l_small_left,t_small,((2/inv_dx)*gamma_left),b_small)
    x_left = np.linalg.solve(a_left, -b_left)
    #set interface_points_left with x_left 
    #start with right room
    #Update r_small_left with help from x_middle
    gamma_right = np.zeros(inv_dx-1)
    for i in xrange(gamma_right.size):
        gamma_right[i] = gamma(x_right_old[i], temp_x[:inv_dx-1,-1][i])
    b_right = make_B_vector(l_small_right,t_small, ((2/inv_dx)*gamma_right),b_small)
    x_right = np.linalg.solve(a_right,-b_right)
   
    #Update interface_points_right with x_right
    #Update x_right_old, x_left_old
   
    interface_points_left = x_left.reshape(inv_dx-1,inv_dx)[:,inv_dx-1]
    interface_points_right = x_right.reshape(inv_dx-1,inv_dx)[:,0]
    
    x_left_old = x_left.reshape(inv_dx-1,inv_dx)[:,1]
    x_right_old = x_right.reshape(inv_dx-1, inv_dx)[:,1]

    #done with this iteration

# Vectors: x_middle, x_right (höger rum), x_left (vänster rum)
# x_middle.reshape(5,2) för inv_dx = 3
# x_right och x_left har olika reshapes

# För stora rummet: t_middle l_middle, r_middle, b_middle utöver x_middle
# För lilla rummet till vänster: t_small, b_small, l_small_left, l_small_right
# För lilla rummet till höger: t_small, b_small, r_small_left, r_small_right ut

#print(x_middle.reshape(inv_dx*2 - 1,inv_dx -1))
#new_vec = np.ones([10,10])
#imgplot = plt.imshow(x_middle.reshape(5,2))
#print(b_middle)



