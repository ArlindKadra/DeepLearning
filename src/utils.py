import numpy as np


def feature_normalization(x, categorical=None):
    """
    Do feature normalization on the input by removing the mean
    and dividing by the standard deviation.
    Parameters
    ----------
    x : np.array,
        Input.
    categorical : np.array, optional
        Boolean array which represents categorical features.
    Returns
    -------
    np.array
        Normalized input.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    if categorical is None:
        # normalize all features
        x = (x - mean) / std
    else:
        for row in range(0, x.shape[0]):
            for column in range(0, x.shape[1]):
                # only normalize features which are not categorical
                if not categorical[column]:
                    x[row, column] = (x[row, column] - mean[column]) / std[column]

    return x, mean, std


def contains_nan(x):

    result = np.isnan(x)
    if True in result:
        return True
    else:
        return False
