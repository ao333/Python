import numpy as np

def PrintMatrix(matrix):
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix.dtype == np.int32:
                print matrix[row,col],
            else:
                print "%0.3f" % matrix[row,col],
            print "\t",
        print ""
