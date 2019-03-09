import logging
import autosklearn.classification
from sklearn.metrics import accuracy_score
import os

class RandomForest(object):
    # Class which resembles an auto-sklearn random forest classifier

    def __init__(self, x, y, categorical_indicator, split_indices, task_id):

        super(RandomForest, self).__init__()

        logger = logging.getLogger(__name__)
        # Create feature type list indicator and run autosklearn
        self.feat_type = ['Categorical' if feature else 'Numerical'
                     for feature in categorical_indicator]

        train_split_indices = split_indices[0]
        test_split_indices = split_indices[1]

        self.fitted_instance = None
        self.x_train = x[train_split_indices]
        self.y_train = y[train_split_indices]
        self.x_test = x[test_split_indices]
        self.y_test = y[test_split_indices]

        self.task_id = task_id

        logger.info("Started training AutoSklearn with %s estimator"
                    % RandomForest.get_name())

    def build(self, time, run_id):

        validation_policy = {'cv': {'folds': 5, 'shuffle': True}}
        main_path = os.path.expanduser(os.path.join('autosklearn_exp', self.get_name(), '%d' % self.task_id))
        output_path = os.path.join(main_path, 'output', str(run_id))
        tmp_path = os.path.join(main_path, 'tmp', str(run_id))

        return autosklearn.classification.AutoSklearnClassifier(
            include_estimators=["random_forest"],
            time_left_for_this_task=time,
            ensemble_size=1,
            output_folder=output_path,
            delete_output_folder_after_terminate=True,
            tmp_folder=tmp_path,
            delete_tmp_folder_after_terminate=True,
            resampling_strategy='cv',
            resampling_strategy_arguments=validation_policy,
            initial_configurations_via_metalearning=0,
            n_jobs=20
        )

    def train(self, time, run_id):

        random_forest = self.build(time, run_id)
        random_forest.fit(self.x_train, self.y_train, feat_type=self.feat_type)
        random_forest.refit(self.x_train, self.y_train)
        self.fitted_instance = random_forest

    def predict(self):

        logger = logging.getLogger(__name__)
        if self.fitted_instance is None:
            raise ValueError
        else:
            y_prediction = self.fitted_instance.predict(self.x_test)
            model_accuracy = accuracy_score(self.y_test, y_prediction)

        logger.info('Model accuracy: %f' % model_accuracy)

        return model_accuracy

    @staticmethod
    def get_name():

        return "random_forest"

