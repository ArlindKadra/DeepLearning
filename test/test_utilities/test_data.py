import unittest
from collections import Counter

import numpy as np

import src.model as model
from utilities.data import (
    calculate_stat,
    determine_input_sets,
    determine_stratified_val_set,
    feature_normalization
)


class TestData(unittest.TestCase):

    def setUp(self):

        model.Loader(23)

        self.x, self.y, _ = model.get_dataset()
        self.train_indices, self.test_indices = \
            model.get_split_indices()

    def test_determine_input_sets(self):

        nr_examples = 1000
        # get the input indices
        train, val, test = determine_input_sets(nr_examples)
        # join input indices
        examples = np.concatenate((train, val, test))
        unique_examples = set(examples)
        # should be the number of examples
        # if everything was done correctly
        self.assertEqual(len(unique_examples), nr_examples)

    def test_calculate_stat(self):

        x = [[5, 6], [7, 8]]
        exp_mean = [6, 7]
        exp_std = [1, 1]
        mean, std = calculate_stat(x)
        # mean and stf
        # should be feature wise
        self.assertTrue(np.alltrue(np.equal(exp_mean, mean)))
        self.assertTrue(np.alltrue(np.equal(exp_std, std)))

    def test_feature_normalization(self):

        x = np.asarray([[4, 5, 1], [6, 9, 3]])
        categorical_ind = [False, False, True]
        mean, std = calculate_stat(x)
        data = feature_normalization(
            x,
            mean,
            std,
            categorical_ind
        )
        expected_result = [[-1, -1, 1], [1, 1, 3]]
        self.assertTrue(np.alltrue((np.equal(data, expected_result))))

    def test_determine_stratified_val_set(self):

        nr_folds = 10
        # get the input indices
        y_train_data = self.y[self.train_indices]
        train_indices, val_indices = \
            determine_stratified_val_set(
                self.x[self.train_indices],
                self.y[self.train_indices]
            )

        y_train = y_train_data[train_indices]
        y_val = y_train_data[val_indices]
        instances_train = Counter()
        instances_val = Counter()
        for label in y_train:
            instances_train[label] += 1
        for label in y_val:
            instances_val[label] += 1
        # for each class the number of instances
        # between folds, differs at most with
        # the number of folds - 1
        for label in y_train:
            self.assertTrue(abs((instances_train[label] / 9) - instances_val[label]) <= nr_folds - 1)