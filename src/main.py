from src.models import fcresnet
import src.utils as utils


import openml
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(level=logging.DEBUG)


benchmark_suite = openml.study.get_study("99", "tasks")
task_id = random.choice(list(benchmark_suite.tasks))
dataset = openml.tasks.get_task(task_id).get_dataset()

x, y, categorical = dataset.get_data(target=dataset.default_target_attribute,
                                     return_categorical_indicator=True)

# For the moment, do not deal with input that contains NaN values
if utils.contains_nan(x):
    raise ValueError("Input contains Nan values")

if True in categorical:
    x, mean, std = utils.feature_normalization(x, categorical)
    enc = OneHotEncoder(categorical_features=categorical, dtype=np.float32)
    x = enc.fit_transform(x).todense()
else:
    x, mean, std = utils.feature_normalization(x, None)


config_space = fcresnet.get_config_space()
config = config_space.sample_configuration(1)

# initially a hardcoded number of folds
nr_folds = 10
kf = KFold(n_splits=nr_folds)

results = list()

for train_indices, test_indices in kf.split(x):
    x_train, y_train = x[train_indices], y[train_indices]
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 1 / (nr_folds - 1))
    x_test, y_test = x[test_indices], y[test_indices]
    results.append(fcresnet.train(config, 2, x_train, y_train, x_test, y_test))

print(np.mean(results))
