import numpy as np

'''
set - the 2D matrix to split row-wise
i - the start index of the segment
k - the total number of segments to split
n - the number of segments to take out
'''
def SplitSet(set, i, k = 10, n = 1):
    if i >= k:
        return False
    length = set.shape[0]
    foldSize = length / k
    splitStart = i * foldSize;
    if (i + n) >= k:
        # ensure we can handle wrap-around
        splitEnd = (i + n) - k
        train1 = set[splitEnd:splitStart, :]
        train2 = np.ndarray(shape=(0, train1.shape[1]), dtype=set.dtype)
        validate = np.append(set[splitStart:, :], set[:splitEnd, :], axis = 0)
    else:
        splitEnd = (i + n) * foldSize;
        train1, validate, train2 = set[:splitStart, :], set[splitStart:splitEnd, :], set[splitEnd:, :]

    train = np.append(train1, train2, axis = 0)
    return train, validate
