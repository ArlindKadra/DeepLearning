from utilities import data
import logging
import openml


_x = None
_y = None
_categorical = None
_mean = None
_std = None
_task_id = None


class Loader(object):

    def __init__(self, task_id=3):

        global _x, _y, _categorical, _mean, _std, _task_id
        logger = logging.getLogger(__name__)
        _task_id = task_id
        dataset = openml.tasks.get_task(_task_id).get_dataset()

        x, y, categorical = dataset.get_data(
            target=dataset.default_target_attribute,
            return_categorical_indicator=True
        )

        logger.info("Data from task id %d Loaded", _task_id)

        # TODO use mean or other ways to fill the missing values.
        # For the moment, do not deal with input that contains NaN values
        if data.contains_nan(x):
            # TODO switch to logger exception
            logger.error('Input contains NaN values')
            raise ValueError("Input contains NaN values")

        _x = x
        _y = y
        _categorical = categorical


def get_dataset():

    return _x, _y, _categorical


def get_mean():

    return _mean


def get_std():

    return _std


def get_task_id():

    return _task_id
