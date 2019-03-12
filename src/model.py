from utilities import data
import logging
import openml

_x = None
_y = None
_categorical = None
_task_id = None
_train_indices = None
_test_indices = None


class Loader(object):

    def __init__(self, task_id=3):

        global _x, _y, _categorical, _task_id, _train_indices, _test_indices
        logger = logging.getLogger(__name__)
        _task_id = task_id
        task = openml.tasks.get_task(_task_id)
        dataset = task.get_dataset()

        x, y, categorical = dataset.get_data(
            target=dataset.default_target_attribute,
            return_categorical_indicator=True
        )

        _train_indices, _test_indices = task.get_train_test_split_indices()

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


def get_task_id():

    return _task_id


def get_split_indices():

    return _train_indices, _test_indices
