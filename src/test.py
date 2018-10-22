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

import numpy as np

a = [
    [1,2,3,4,6],
    [1,2,3,4,5],
]

print(np.mean(a, axis=0))