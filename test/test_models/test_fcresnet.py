import unittest

from src.models.fcresnet import FcResNet
import src.model as model
from src.utilities.search_space import get_fixed_fcresnet_config
from src.utilities.data import determine_feature_type


class TestFcResNet(unittest.TestCase):

    def setUp(self):

        model.Loader(3)

        self.x, self.y, self.categorical \
            = model.get_dataset()

    def test_fcnet_no_reg(self):

        feature_type = determine_feature_type(self.categorical)
        nr_features = self.x.shape[1]

        config = get_fixed_fcresnet_config(
            nr_features,
            feature_type,
            super_blocks=2,
            num_res_blocks=4,
            activate_dropout='Yes'
        ).sample_configuration()

        network = FcResNet(
            config,
            self.x.shape[1],
            len(set(self.y))
        )

        print(network)

        total_params = sum(p.numel() for
                           p in network.parameters() if p.requires_grad)

        weights = self.x.shape[1] * 64

        # number of biases input layer
        weights += 64
        # number of weights between remaining layers
        weights += 64 * 64 * 4
        # number of weights between output layer and last layer
        weights += 64 * len(set(self.y))
        # number of biases - 4 layers + output layer
        weights += 64 * 4 + len(set(self.y))
        self.assertEqual(weights, total_params)

    def test_fcresnet_all_reg(self):

        x, y, categorical = model.get_dataset()
        feature_type = determine_feature_type(categorical)
        nr_features = x.shape[1]

        config = get_fixed_fcresnet_config(
            nr_features,
            feature_type,
            num_res_blocks=2,
            activate_dropout='Yes',
            activate_batch_norm='Yes',
            activate_mixout='Yes',
            activate_shake_shake='Yes',
            activate_weight_decay='Yes'
        ).sample_configuration()

        network = FcResNet(
            config,
            self.x.shape[1],
            len(set(self.y))
        )

        total_params = sum(p.numel() for
                           p in network.parameters() if p.requires_grad)

        # number of weights between remaining layers
        weights = 64 * 64 * 4
        # number of batchnorm weights in res blocks
        # res part
        weights += 64 * 2 * 4
        # number of biases - 4 layers
        weights += 64 * 4
        # because of shake shake we have double
        # of the above
        weights *= 2
        # number of batchnorm weights in skip connec
        # weights += 64 * 2 * 2
        # number of weights between output layer and last layer
        weights += 64 * 2
        # number of biases output layer
        weights += 2
        # batch norm weights input layer
        weights += 64 * 2
        # number of biases for input layer
        weights += 64
        # number of weights from input layer

        weights += x.shape[1] * 64

        self.assertEqual(weights, total_params)


if __name__ == '__main__':
    unittest.main()
