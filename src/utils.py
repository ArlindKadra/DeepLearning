import models.fcresnet

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import logging
import os


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


def setup_logging(log_file, level=logging.INFO):
    # TODO Read main logs dir from configuration
    main_logs_dir='logs'
    root = logging.getLogger()
    root.setLevel(level)
    format = '%(asctime)s, %(process)-6s %(levelname)-5s %(module)s: %(message)s'

    date_format = '%H:%M:%S'

    f = logging.Formatter(format, date_format)
    ch = logging.StreamHandler()
    ch.setFormatter(f)
    root.addHandler(ch)

    os.makedirs(os.path.join(main_logs_dir), exist_ok=True)
    log_file = os.path.join(main_logs_dir, '{}.log'.format(log_file))
    fh = logging.FileHandler(log_file)
    fh.setFormatter(f)
    root.addHandler(fh)


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


def cross_validation(nr_epochs, x, y, config, nr_folds=10):
    """Use cross validation to train the network.

    Args:
        nr_epochs: Number of epochs to train the network on.
        x: Input.
        y: Labels.
        config: ConfigSpace configuration
        nr_folds: Number of cross-validation folds.

    Returns:
        A float value which shows the average accuracy
        achieved while training with cross-validation.
    """

    val_loss_epochs = np.zeros(nr_epochs)
    test_loss = 0
    test_accuracy = 0

    # Shuffle data before, otherwise the results on some tasks were confusing.
    # Validation had similiar loss to the training data while test data had a very high one.
    # np.random.shuffle(x)

    kf = KFold(n_splits=nr_folds, shuffle=True)

    for train_indices, test_indices in kf.split(x):
        x_train, y_train = x[train_indices], y[train_indices]
        x_train, x_validation, \
            y_train, y_validation = \
            train_test_split(x_train, y_train, test_size=1 / (nr_folds - 1))
        x_test, y_test = x[test_indices], y[test_indices]
        output = models.fcresnet.train(config, nr_epochs, x_train, y_train, x_validation, y_validation, x_test, y_test)
        val_loss_epochs = np.add(val_loss_epochs, output['validation'])
        test_loss += output['test'][0]
        test_accuracy += output['test'][1]

    # average the values over the folds
    val_loss_epochs = val_loss_epochs / nr_folds
    test_loss = test_loss / nr_folds
    test_accuracy = test_accuracy / nr_folds
    result = {'test_loss': test_loss, 'test_accuracy': test_accuracy,
              'val_loss': list(val_loss_epochs)}
    return result


def mixup_criterion(y_a, y_b, lam):

    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
