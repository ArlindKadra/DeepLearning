import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def feature_normalization(x, categorical=None):
    """Do feature normalization on the input

    Normalize the input by removing the mean and
    dividing by the standard deviation.

    Args:
        x: Input.
        categorical: Boolean array which represents categorical features.

    Returns:
        Normalized input tensor.
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


def cross_validation(nr_epochs, x, y, network, config, nr_folds=10):
    """Use cross validation to train the network.

    Args:
        nr_epochs: Number of epochs to train the network on.
        x: Input.
        y: Labels.
        network: Pytorch network.
        config: ConfigSpace configuration
        nr_folds: Number of cross-validation folds.

    Returns:
        A float value which shows the average accuracy
        achieved while training with cross-validation.
    """

    accuracy_results = list()
    loss_results = list()
    kf = KFold(n_splits=nr_folds)

    for train_indices, test_indices in kf.split(x):
        x_train, y_train = x[train_indices], y[train_indices]
        x_train, x_validation, \
            y_train, y_validation = \
            train_test_split(x_train, y_train, test_size=1 / (nr_folds - 1))
        x_test, y_test = x[test_indices], y[test_indices]
        accuracy, loss = network.train(config, nr_epochs, x_train, y_train, x_test, y_test)
        accuracy_results.append(accuracy)
        loss_results.append(loss)
    return loss_results, accuracy_results
