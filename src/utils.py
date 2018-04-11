import numpy as np


def feature_normalization(x):

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = (x - mean) / std

    return x, mean, std


def contains_nan(x):

    result = np.isnan(x)
    if True in result:
        return True
    else:
        return False

