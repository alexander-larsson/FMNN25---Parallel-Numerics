#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Attempt of solving using Forward and Backward differences
# instead of Central Differences

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
            A[bi][bi] = -3
        elif side == "R":
            bi = (size - 2) + (size-1)*i
            A[bi][bi-1] = 2
            A[bi][bi] = -3
    return A

def make_A_matrix_big_room(size):
    """
    Creates the A matrix for the big room
    Parameters:
    size: tuple on the form (m,n)
    """
    # Måste lägga in -3 på rätt ställe här
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

print make_A_matrix_small_room(4,"L")
