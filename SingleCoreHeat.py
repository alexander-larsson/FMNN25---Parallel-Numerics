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
R1[-1,:] = np.ones(n_s) * normalT
R1[:,0] = np.ones(n_s) * heaterT

# This is what R1 looks like
# [[ 40.  15.  15.  15.]
#  [ 40.   0.   0.   0.]
#  [ 40.   0.   0.   0.]
#  [ 40.  15.  15.  15.]]

R3 = np.zeros((n_s,n_s))

R3[0,:] = np.ones(n_s) * normalT
R3[-1,:] = np.ones(n_s) * normalT
R3[:,-1] = np.ones(n_s) * heaterT

# This is what R3 looks like
# [[ 15.  15.  15.  40.]
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

# WIP DO NOT TOUCH THIS

# Will write algorithm to convert these room matrices into equation systems
# that we solve to get the answer

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

    # This for loops puts all the ones in the matrix
    for i in xrange(rows):
        for j in xrange(cols):
            if room_matrix[i][j] == 0: #Means it is an unknown
                nearby = [(i+1,j),(i,j+1),(i-1,j),(i,j-1)]
                for k in xrange(len(nearby)):
                    ni,nj = nearby[k]
                    if ni < rows and nj < cols: # Still inside matrix
                        if room_matrix[ni][nj] == 0: # Nearby element is unknown
                            ## Stoppa in etta i matrisen A
                            ## Detta är sjuuuukt klurigt med indexen :D
                        else:
                            b[ui] -= room_matrix[ni][nj]
                ui += 1

    return A,b
