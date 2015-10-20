#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

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

def update_B_vector_big_room(b,interface_points_left,interface_points_rigth):

    ##  Insert the calculated points in b

    return b

def update_B_vector_small_room(b,interface_side,guess):
    """
    Parameters:
    b: the vector created by running _make_B_matrix_
    neumann_side: the string "L" or "R" to specify where the interface is
    """



#print make_A_matrix_small_room(5,side="L")
#print make_A_matrix_big_room((7,4))
l = np.ones(5)*15
t = np.ones(3)*40
b = np.ones(3)*5
r = np.ones(5)*15
print make_B_vector_big_room(l,t,r,b)
