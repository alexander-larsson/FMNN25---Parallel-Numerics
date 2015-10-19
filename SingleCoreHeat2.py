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
    for i in xrange(m-1,A_size):
        A[i-m+1][i] = 1
        A[i][i-m+1] = 1
    return A

#print make_A_matrix_small_room(5,side="L")
print make_A_matrix_big_room((7,4))
