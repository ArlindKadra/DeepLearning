import utils

import logging
import openml
import numpy as np
from sklearn.preprocessing import OneHotEncoder

_x = None
_y = None
_categorical = None
_mean = None
_std = None
_task_id = None


class Loader(object):

    def __init__(self, task_id=3, torch=True):

        global _x, _y, _categorical, _mean, _std, _task_id
        logger = logging.getLogger(__name__)
        dataset = openml.tasks.get_task(_task_id).get_dataset()

        x, y, categorical = dataset.get_data(target=dataset.default_target_attribute,
                                             return_categorical_indicator=True)
        logger.info("Data Loaded")

        # TODO use mean or other ways to fill the missing values.
        # For the moment, do not deal with input that contains NaN values
        if utils.contains_nan(x):
            # TODO switch to logger exception
            logger.error('Input contains NaN values')
            raise ValueError("Input contains NaN values")

        if True in categorical:
            x, mean, std = utils.feature_normalization(x, categorical)
            if torch:
                enc = OneHotEncoder(categorical_features=categorical, dtype=np.float32)
                x = enc.fit_transform(x).todense()
        else:
            x, mean, std = utils.feature_normalization(x, None)

        logger.info("Data normalized")

        _x = x
        _y = y
        _task_id = task_id
        _categorical = categorical
        _mean = mean
        _std = std

def get_dataset():

    return _x, _y, _categorical

def get_mean():

    return _mean

def get_std():

    return _std

def get_task_id():

    return _task_id

