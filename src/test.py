import numpy as np
import openml


dataset = openml.tasks.get_task(3).get_dataset()

x, y, categorical = dataset.get_data(
    target=dataset.default_target_attribute,
    return_categorical_indicator=True
)

labels = np.zeros((y.shape[0], max(y) + 1))
labels[np.arange(y.shape[0]), y] = 1

b = np.ones((3, 2))

b = b * 7
c = np.ones((3, 2))
b = b + c

a = np.arange(1, 9)
a = a.reshape(4, 2)

b = [0, 2, 3]

# print(a[b])
d = np.arange(1, 9)
print(np.random.permutation(d))
print(d)