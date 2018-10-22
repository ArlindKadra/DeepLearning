import numpy as np
from sklearn.model_selection import train_test_split


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

    return x, mean, std


def calculate_stat(x):

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    return mean, std


def shuffle_data(x, y):

    indices = np.arange(0, len(x))
    np.random.seed(11)
    shuffled_indices = np.random.permutation(indices)
    return x[shuffled_indices], y[shuffled_indices]


def separate_input_sets(x, y):

    x_train, x_test, \
        y_train, y_test = \
        train_test_split(x, y, test_size=1 / 10, random_state=69)
    x_train, x_val, \
        y_train, y_val = \
        train_test_split(x_train, y_train, test_size=1 / 9)

    examples = {'train': x_train, 'val': x_val, 'test': x_test}
    labels = {'train': y_train, 'val': y_val, 'test': y_test}

    return examples, labels


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

