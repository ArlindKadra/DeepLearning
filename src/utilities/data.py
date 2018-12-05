import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold


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


def calculate_class_weights(y_train):

    return compute_class_weight(
        'balanced',
        np.unique(y_train),
        y_train
    )


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


def determine_stratified_val_set(x_train, y_train, nr_folds=10):

    skf = StratifiedKFold(n_splits=nr_folds)
    train_indices = None
    validation_indices = None
    for train_set, validation_set in skf.split(x_train, y_train):
        train_indices = train_set
        validation_indices = validation_set
        break
    return train_indices, validation_indices


def determine_feature_type(categorical):

    if True in categorical:
        if False in categorical:
            feature_types = 'mixed'
        else:
            feature_types = 'categorical'
    else:
        feature_types = 'numerical'

    return feature_types


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
