import openml
import random
import fcresnet
from fcresnet import FcResNet
from sklearn import preprocessing

benchmark_suite = openml.study.get_study("99", "tasks")
task_id = random.choice(list(benchmark_suite.tasks))
dataset = openml.tasks.get_task(task_id).get_dataset()
x, y, categorical = dataset.get_data(target=dataset.default_target_attribute,
                                     return_categorical_indicator=True)
#enc = preprocessing.OneHotEncoder(categorical_features=categorical)
#print(enc)
print("bla")
#x = enc.fit_transform(x)
#print(x)
config_space = fcresnet.get_config_space()
config = config_space.sample_configuration(1)
fcresnet.train(config, 100, x, y)
