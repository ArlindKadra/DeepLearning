import numpy as np


def feature_normalization(x, mean, std, categorical=None):
    """Do feature normalization on the input

    Normalize the input by removing the mean and
    dividing by the standard deviation.

    Args:
        x: Input.
        mean: mean
        std: standard deviation.
        categorical: Boolean array which represents categorical features.

    Returns:
        Normalized input tensor.
    """

    if categorical is None:
        # normalize all features
        x = (x - mean) / std
    else:
        for row in range(0, x.shape[0]):
            for column in range(0, x.shape[1]):
                # only normalize features which are not categorical
                if not categorical[column]:
                    x[row, column] = (x[row, column] - mean[column]) / std[column]

    return x


def calculate_stat(x):

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    return mean, std


def determine_input_sets(nr_examples):

    # Generate list with example indexes
    index_slice = int(1/10 * nr_examples)
    indices = np.arange(0, nr_examples)
    np.random.seed(11)
    shuffled_indices = np.random.permutation(indices)
    # determine the indices for each set
    test = shuffled_indices[0:index_slice]
    validation = shuffled_indices[index_slice:2 * index_slice]
    train = shuffled_indices[2 * index_slice:]

    return train, validation, test


def contains_nan(x):
    """Validate the tensor.

    Check whether the tensor contains NaN values.

    Args:
        x: tensor.

    Returns:
        A boolean showing whether the input contains NaN or not.
    """
    result = np.isnan(x)
    if True in result:
        return True
    else:
        return False
