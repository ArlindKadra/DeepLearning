import logging
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    filename='network.log', level=logging.INFO)

from src.models import fcresnet
from src.models import random_forest
import src.utils as utils

import openml
import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder



def main():

    logging.info('Experiment started')
    benchmark_suite = openml.study.get_study("99", "tasks")
    task_id = random.choice(list(benchmark_suite.tasks))
    dataset = openml.tasks.get_task(task_id).get_dataset()

    x, y, categorical = dataset.get_data(target=dataset.default_target_attribute,
                                         return_categorical_indicator=True)

    # For the moment, do not deal with input that contains NaN values
    if utils.contains_nan(x):
        logging.error('Input contains NaN values')
        raise ValueError("Input contains NaN values")

    # random_forest.train(x, y, categorical)

    if True in categorical:
        x, mean, std = utils.feature_normalization(x, categorical)
        enc = OneHotEncoder(categorical_features=categorical, dtype=np.float32)
        x = enc.fit_transform(x).todense()
    else:
        x, mean, std = utils.feature_normalization(x, None)

    #config_space = fcresnet.get_config_space()
    #config = config_space.sample_configuration(1)
    #result = utils.cross_validation(x, y, fcresnet.FcResNet, config)
    #logging.info('Cross Validation loss: %.3f, accuracy %.3f', result[0], result[1])


if __name__ == '__main__':
    main()
