import numpy as np

def CountOccurrence(y, num_classes):
    occurrence = np.zeros(num_classes, np.int32)
    for i in range(0, y.shape[0]):
        occurrence[y[i,0]-1] += 1
    return occurrence
