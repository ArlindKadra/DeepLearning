import utils

import logging
import openml
import numpy as np
from sklearn.preprocessing import OneHotEncoder

_x, _y, _categorical, _mean, _std, _task_id


class Loader(object):

    def __init__(self, torch=True):

        global _x, _y, _categorical, _mean, _std, _task_id
        logger = logging.getLogger(__name__)
        # TODO parse the value from the config file
        _task_id = 167141
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
        _categorical = categorical
        _mean = mean
        _std = std

def get_dataset(self):

    return _x, _y, _categorical

def get_mean(self):

    return _mean

def get_std(self):

    return _std

def get_task_id(self):

    return _task_id
