from __future__ import division
import numpy as np

# WIP DO NOT TOUCH THIS
# WIP DO NOT TOUCH THIS
# WIP DO NOT TOUCH THIS

# Example code for delta = 1/3

inv_delta = 3 #Easier to use than "real" delta (1/3)

heaterT = 40
windowT = 5
normalT = 15

# Rooms counted left to right

# Room 1 matrix

n_s = 4 # Size of small rooms

R1 = np.zeros((n_s,n_s))

R1[0,:] = np.ones(n_s) * normalT
R1[-1,:] = np.ones(n_s) * windowT
R1[:,0] = np.ones(n_s) * heaterT

# This is what R1 looks like
# [[ 40.  15.  15.  15.]
#  [ 40.   0.   0.   0.]
#  [ 40.   0.   0.   0.]
#  [ 40.   5.   5.   5.]]

R3 = np.zeros((n_s,n_s))

R3[0,:] = np.ones(n_s) * heaterT
R3[-1,:] = np.ones(n_s) * normalT
R3[:,-1] = np.ones(n_s) * heaterT

# This is what R3 looks like
# [[ 40.  40.  40.  40.]
#  [  0.   0.   0.  40.]
#  [  0.   0.   0.  40.]
#  [ 15.  15.  15.  40.]]

R2 = np.zeros((7,4))

R2[0:4,0] = np.ones(n_s) * normalT
R2[3:7,-1] = np.ones(n_s) * normalT
R2[0,:] = np.ones(n_s) * heaterT
R2[-1,:] = np.ones(n_s) * windowT

# R2 looks like this
# [[ 40.  40.  40.  40.]
#  [ 15.   0.   0.   0.]
#  [ 15.   0.   0.   0.]
#  [ 15.   0.   0.  15.]
#  [  0.   0.   0.  15.]
#  [  0.   0.   0.  15.]
#  [  5.   5.   5.   5.]]

# Update values at interface points like this

# R2[4:6,0] = np.array([1,2])
# R2[1:3,-1] = np.array([3,4])

# Then R2 looks like this
# [[ 40.  40.  40.  40.]
#  [ 15.   0.   0.   3.]
#  [ 15.   0.   0.   4.]
#  [ 15.   0.   0.  15.]
#  [  1.   0.   0.  15.]
#  [  2.   0.   0.  15.]
#  [  5.   5.   5.   5.]]

def make_small_room_eq_sys(room_matrix):
    """
    Calculates A and b from a room matrix
    """
    rows, cols = room_matrix.shape
    u_rows = rows-1
    u_cols = cols-2

    # N unknowns in the equation system
    n = (u_rows)*(u_cols)
    b = np.zeros(n)
    A = np.diag(np.ones(n)*-4)

    ui = 0 # Unknown index

    # Magic check:
    if room_matrix[1][0] == 0:
        bi,bj = 1,0 # Right small room
    else:
        bi,bj = 1,1 # Left small room

    inner_rows = rows - 2
    inner_cols = cols - 1

    # This for loops puts all the ones in the matrix
    for i in xrange(bi,bi+inner_rows):
        for j in xrange(bj,bj+inner_cols):
            nearby = [(i+1,j),(i,j+1),(i-1,j),(i,j-1)]
            for ni,nj in nearby:
                if ni < rows and nj < cols: # Still inside matrix
                    if room_matrix[ni][nj] == 0: # Nearby element is unknown
                        A[ui][(ni-i)*inner_cols + nj - i] = 1
                    else:
                        b[ui] -= room_matrix[ni][nj]
            ui += 1

    return A,b

A,b = make_small_room_eq_sys(R1)

print A
print b
