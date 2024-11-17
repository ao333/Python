import numpy as np

def MajorityValue(binary_targets):
    num_ones = np.count_nonzero(binary_targets)
    num_zeros = len(binary_targets) - num_ones
    if num_ones > num_zeros:
        return 1
    return 0
