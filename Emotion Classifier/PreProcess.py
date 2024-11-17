import numpy as np

def PreProcess (y, targetV):
    newTarget = np.zeros(y.shape[0], dtype=np.int32)
    for i in range(0, y.shape[0]):
        if y[i] == targetV:
            newTarget[i] = 1

    return newTarget.reshape((-1,1))
