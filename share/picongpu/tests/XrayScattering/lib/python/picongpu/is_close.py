import numpy as np


def is_close(input1, input2, abs_tolerance, threshold, rel_tolerance):
    assert input1.dtype.type is input2.dtype.type
    diff = np.abs(input1 - input2)
    check0 = np.minimum(np.abs(input1), np.abs(input2)) < threshold
    check1 = diff < abs_tolerance
    check2 = diff < rel_tolerance * np.maximum(np.abs(input1), np.abs(input2))
    return np.all(np.logical_or(np.logical_and(check0, check1), check2))
