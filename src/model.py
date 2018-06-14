import utils

import logging
import openml
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Loader(object):

    def __init__(self, torch=True):

        logger = logging.getLogger(__name__)
        # TODO parse the value from the config file
        task_id = 167141
        dataset = openml.tasks.get_task(task_id).get_dataset()

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
        self._x = x
        self._y = y
        self._categorical = categorical
        self._mean = mean
        self._std = std

    def get_dataset(self):

        return self._x, self._y, self._categorical

    def get_mean(self):

        return self._mean

    def get_std(self):

        return self._std
