import logging
import autosklearn.classification
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def train(x_train, y_train, categorical_indicator):

    # Create feature type list indicator and run autosklearn
    feat_type = ['Categorical' if feature else 'Numerical'
                 for feature in categorical_indicator]
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=1 / 10)
    svm = Svm().build()
    svm.fit(x_train, y_train, feat_type=feat_type)
    y_prediction = svm.predict(x_test)
    logging.info('SVM accuracy score: %f %%', accuracy_score(y_test, y_prediction))
    logging.info('SVM loss: %f %%', log_loss(y_test, y_prediction))


class Svm(object):
    # Class which resembles an auto-sklearn svm classifier

    def __init__(self):

        super(Svm, self).__init__()

    def build(self):

        return autosklearn.classification.AutoSklearnClassifier(
            include_estimators=["libsvm_svc"], ensemble_size=1,
            initial_configurations_via_metalearning=0)
