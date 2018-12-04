"""
import openml


task = openml.tasks.get_task(3)
x, y = task.get_X_and_y()
print("Input")
print(len(x))
_, folds, samples = task.get_split_dimensions()
for fold in range(1):
    train_indeces, test_indeces = task.get_train_test_split_indices(fold=fold)
    print(y[test_indeces])
    print(len(test_indeces))
"""
"""
import re
extra_match = re.search(r"\D+\d+(\d|\])*$", "1111111_[1]")
if extra_match:
    extra_part = extra_match.group(0)
    print(extra_part)
    run_id = re.sub(extra_part, "", "1111111_[1]")
print(run_id)
"""
from sklearn.model_selection import StratifiedKFold
import numpy as np

x = np.array([6, 6, 6, 6, 6, 6])
y = np.array([0, 2 , 2 , 1, 1, 0])
skf = StratifiedKFold(n_splits=2)
for train_set, validation_set in skf.split(x, y):
    train_indices = train_set
    validation_indices = validation_set
    break
print(train_indices)
print(validation_indices)