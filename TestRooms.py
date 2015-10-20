import numpy as np



def createMatrix(size):
    n = size
    #Create matrix
    matrix = np.eye(n)-5*np.eye(n)
#room_one = np.ones((n,n))
#diagonal = np.flipud(diagonal)
#room_one = room_one - diagonal
#room_one = room_one - diagonal

    
    for i in range(n):
        for k in range(n):

    print(matrix)

createMatrix(4)


