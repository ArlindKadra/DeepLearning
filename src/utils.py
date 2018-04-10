import numpy as np


def feature_normalization(x):

    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)

    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            x[i, j] = (x[i, j] - mean[j]) / std[j]

    return x, mean, std