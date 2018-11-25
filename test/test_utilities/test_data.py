import unittest
import numpy as np

from utilities.data import (
    calculate_stat,
    determine_input_sets,
    feature_normalization
)


class TestData(unittest.TestCase):

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
        input = feature_normalization(
            x,
            mean,
            std,
            categorical_ind
        )
        expected_result = [[-1, -1, 1], [1, 1, 3]]
        self.assertTrue(np.alltrue((np.equal(input, expected_result))))
