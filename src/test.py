import numpy as np
import openml


dataset = openml.tasks.get_task(3).get_dataset()

x, y, categorical = dataset.get_data(
    target=dataset.default_target_attribute,
    return_categorical_indicator=True
)

print(y)

b = [[1, 2, 3], [4, 6, 5], [7,8, 9], [10, 11, 12]]
print(np.argmax(b, axis=1))