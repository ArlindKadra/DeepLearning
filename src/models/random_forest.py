import logging
import numpy as np
import autosklearn.classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def train(x, y, categorical_indicator, task_id):

    # Create feature type list indicator and run autosklearn
    feat_type = ['Categorical' if feature else 'Numerical'
                 for feature in categorical_indicator]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 10)
    random_forest = RandomForest().build(task_id)
    random_forest.fit(x_train, y_train, x_test, y_test, feat_type=feat_type)
    y_prediction = random_forest.predict(x_test)
    print(y_prediction)
    model_accuracy = accuracy_score(y_test, y_prediction)
    print(model_accuracy)
    print(model_accuracy.item())
    logger = logging.getLogger(__name__)
    logger.info('Bla')

class RandomForest(object):
    # Class which resembles an auto-sklearn random forest classifier

    def __init__(self):

        super(RandomForest, self).__init__()


    def build(self, task_id):

        return autosklearn.classification.AutoSklearnClassifier(
            include_estimators=["random_forest"], ensemble_size=1,
            output_folder="autosklearn_exp/%d" % task_id, delete_output_folder_after_terminate=False,
            initial_configurations_via_metalearning=0)
