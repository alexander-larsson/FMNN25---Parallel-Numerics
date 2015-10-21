#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def make_A_matrix_small_room(size,side):
    """
    Creates the A matrix for small rooms(left(L) or right(R))
    Parameters:
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

    #Neumann Conditions
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
    Creates the A matrix for the big room
    Parameters:
    size: tuple on the form (m,n)
    """
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
    Creates the B vector for a room
    Parameters:
    left: left wall vector
    top: top wall vector
    right: right wall vector
    bottom: bottom wall vector
    4 vectors corresponding to the elements on either side of the matrix.
    If the matrix is of size n x m the top and bottom vectors will be n-2.
    The left and right will be m-2.
    For left and right 0 is the top element.
    """
    if not left.shape == right.shape or not top.shape == bottom.shape:
        print "Error, wrong shapes"

    rows = left.size
    cols = top.size
    length = rows*cols
    b = np.zeros(length)

    # Corner 1
    b[0] = top[0]+left[0]

    # rows - 2 element top
    b[1:cols-1] = top[1:-1]

    # Corner 2
    b[cols-1] = top[-1]+right[0]

    # Mid points
    for i in xrange(1,rows-1):
        b[i*cols] = left[i]
        b[i*cols+cols-1] = right[i]

    # Corner 3
    b[length-cols] = bottom[0]+left[-1]

    # rows - 2 element bottom
    b[length-cols+1:-1] = bottom[1:-1]

    # Corner 4
    b[length-1] = bottom[-1]+right[-1]

    return b

def gamma(u_l, u_r):
    """
    Caluclates the gamma value for two points
    Parameters:
    u_l: left point of the value to calculate the gamma in
    u_r: right point of the value to calculate the gamma in
    """
    return (u_l - u_r)/inv_dx*2

#Inverted delta-x value
inv_dx = 3

#Initiliaze vectors,matrix and values for corresponding room down below

#Middle room

#Temperature vectors for the corresponding wall: left,top,bottom,right(Middle room)
l_middle = np.ones(inv_dx*2-1)*15
t_middle = np.ones(inv_dx-1)*40
b_middle = np.ones(inv_dx-1)*5
r_middle = np.ones(inv_dx*2-1)*15


#Creates the A matrix for the middle room, creates the b vector and solves the system
a_middle = make_A_matrix_big_room((inv_dx*2 + 1,inv_dx + 1))
b_vector_middle = make_B_vector(l_middle,t_middle,r_middle,b_middle)
x_middle = np.linalg.solve(a_middle,b_vector_middle)/inv_dx

#Left room

#Temperature vectors for the corresponding wall: left,top,bottom(Left room)
l_small_left = np.ones(inv_dx-1)*40
t_small = np.ones(inv_dx)*15
b_small = np.ones(inv_dx)*15

#Insert interface points in r_small_left
r_small_left = np.ones(inv_dx-1)
for i in xrange(r_small_left.size):
    r_small_left[i] = x_middle[-r_small_left.size*(i+1)]
r_small_left = r_small_left[::-1]

#Creates the A matrix for the left room, creates the b vector and solves the system
a_left = make_A_matrix_small_room(inv_dx +1, "L")
b_left = make_B_vector(l_small_left,t_small,r_small_left,b_small)
x_left = np.linalg.solve(a_left, -b_left)/inv_dx
interface_points_left = x_left.reshape(inv_dx-1,inv_dx)[:,inv_dx-1]


#Right room

#Temperature vectors for the corresponding wall: left,right(Right room)
r_small_right = np.ones(inv_dx-1)*40
l_small_right = np.ones(inv_dx-1)

for i in xrange(r_small_right.size):
    l_small_right[i] = x_middle[l_small_right.size*(i)]

#Creates the A matrix for the right room, creates the b vector and solves the system
a_right = make_A_matrix_small_room(inv_dx +1, "R")
b_right = make_B_vector(l_small_right,t_small,r_small_right,b_small)
x_right = np.linalg.solve(a_left, -b_right)/inv_dx
interface_points_right = x_right.reshape(inv_dx-1,inv_dx)[:,0]

#Saves the interfae points to x_left_old and x_right_old
x_left_old = x_left.reshape(inv_dx-1,inv_dx)[:,1]
x_right_old = x_right.reshape(inv_dx-1, inv_dx)[:,1]

#Iterates 10 times, start in middle room, then go left and then right and so on..
for _ in xrange(1,10):

    #Interface points
    r_middle[:interface_points_left.size] = interface_points_left
    l_middle[-interface_points_right.size:] = interface_points_right

    #Create the b vector for the middle room and solve the system
    b_vector_middle = make_B_vector(l_middle,t_middle,r_middle,b_middle)
    x_middle = np.linalg.solve(a_middle,-b_vector_middle)/inv_dx

    #Left room
    gamma_left = np.zeros(inv_dx-1)
    temp_x = x_middle.reshape(inv_dx*2 -1,inv_dx-1)
    for i in xrange(gamma_left.size):
        gamma_left[i] = gamma(x_left_old[i], temp_x[-(inv_dx-1):,0][i])

    #Create the b vector for the left room and solve the system
    b_left = make_B_vector(l_small_left,t_small,((2/inv_dx)*gamma_left),b_small)
    x_left = np.linalg.solve(a_left, -b_left)/inv_dx


    #Right room
    gamma_right = np.zeros(inv_dx-1)
    for i in xrange(gamma_right.size):
        gamma_right[i] = gamma(temp_x[:inv_dx-1,-1][i],x_right_old[i])

    #Create the b vector for the right room and solve the system
    b_right = make_B_vector(((2/inv_dx)*gamma_right),t_small, r_small_right,b_small)
    x_right = np.linalg.solve(a_right,-b_right)/inv_dx

    #Update interface_points_right with x_right
    #Update x_right_old, x_left_old
    interface_points_left = x_left.reshape(inv_dx-1,inv_dx)[:,inv_dx-1]
    interface_points_right = x_right.reshape(inv_dx-1,inv_dx)[:,0]

    #Saves the solutions x_left and x_right
    x_left_old = x_left.reshape(inv_dx-1,inv_dx)[:,1]
    x_right_old = x_right.reshape(inv_dx-1, inv_dx)[:,1]



# Plotting
middle = x_middle.reshape(inv_dx*2 -1,inv_dx-1)
left = x_left.reshape(inv_dx-1,inv_dx)
right = x_right.reshape(inv_dx-1,inv_dx)

mid_h,mid_w = middle.shape
left_h,left_w = left.shape
right_h,right_w = right.shape

plot_matrix_width = mid_w+left_w+right_w+2
plot_matrix_height = mid_h+2
plot_matrix = np.zeros((plot_matrix_height,plot_matrix_width))


plot_matrix[-inv_dx:-1,1:inv_dx+1] = left
plot_matrix[1:-1,inv_dx+1:2*inv_dx] = middle # tror jag kanske
plot_matrix[1:inv_dx,2*inv_dx:-1] = right
plot_matrix[-1,1:2*inv_dx] = np.ones(2*inv_dx-1)*5
plot_matrix[-inv_dx-1:,0] = np.ones(inv_dx+1)*40
plot_matrix[-inv_dx-1,1:inv_dx+1] = np.ones(inv_dx)*15
plot_matrix[:inv_dx,inv_dx] = np.ones(inv_dx)*15
plot_matrix[0,inv_dx+1:-1] = np.ones(mid_w+right_w)*40
plot_matrix[:inv_dx+1,-1] = np.ones(inv_dx+1)*40
plot_matrix[inv_dx+1:,2*inv_dx] = np.ones(inv_dx)*15
plot_matrix[inv_dx,2*inv_dx:-1] = np.ones(inv_dx)*15

np.set_printoptions(precision=2)
print plot_matrix
heatplot = plt.imshow(plot_matrix)
heatplot.set_cmap('hot')
plt.colorbar()
plt.show()
