import logging
import autosklearn.classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train(x, y, categorical_indicator, task_id):

    logger = logging.getLogger(__name__)
    # Create feature type list indicator and run autosklearn
    feat_type = ['Categorical' if feature else 'Numerical'
                 for feature in categorical_indicator]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 10)
    logger.info("Started training AutoSklearn with %s estimator"
                % RandomForest.get_name())
    random_forest = RandomForest().build(task_id)
    random_forest.fit(x_train, y_train, x_test, y_test, feat_type=feat_type)
    y_prediction = random_forest.predict(x_test)
    model_accuracy = accuracy_score(y_test, y_prediction)
    print(model_accuracy)
    logger.info('Model accuracy: %f' % model_accuracy)


class RandomForest(object):
    # Class which resembles an auto-sklearn random forest classifier

    def __init__(self):

        super(RandomForest, self).__init__()

    def build(self, task_id):

        return autosklearn.classification.AutoSklearnClassifier(
            include_estimators=["random_forest"],
            ensemble_size=1,
            output_folder="autosklearn_exp/%s/%d/output" % (self.get_name(), task_id),
            delete_output_folder_after_terminate=False,
            tmp_folder="autosklearn_exp/%s/%d/tmp" % (self.get_name(), task_id),
            delete_tmp_folder_after_terminate=False,
            initial_configurations_via_metalearning=0
        )

    @staticmethod
    def get_name():

        return "random_forest"
