import openml
import random
import numpy as np
from sklearn.model_selection import KFold
from src.models import fcresnet
import torch
benchmark_suite = openml.study.get_study("99", "tasks")
print("bla")
task_id = random.choice(list(benchmark_suite.tasks))
dataset = openml.tasks.get_task(task_id).get_dataset()

x, y, categorical = dataset.get_data(target=dataset.default_target_attribute,
                                     return_categorical_indicator=True)
# print(enc)
# x = enc.fit_transform(x)

config_space = fcresnet.get_config_space()
config = config_space.sample_configuration(1)

kf = KFold(n_splits=10)
results = list()
for train_indices, test_indices in kf.split(x):
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    results.append(fcresnet.train(config, 2, x_train, y_train, x_test, y_test))
print(np.mean(results))
