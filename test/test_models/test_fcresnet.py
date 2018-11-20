import unittest

from src.models.fcresnet import FcResNet
import src.model as model
from src.utilities.search_space import get_fixed_fcresnet_config


class TestFcResNet(unittest.TestCase):

    def setUp(self):

        model.Loader(3)

        self.x, self.y, _ = model.get_dataset()

    def test_fcnet_no_reg(self):

        config = get_fixed_fcresnet_config(num_res_blocks=2).sample_configuration()

        network = FcResNet(
            config,
            self.x.shape[1],
            len(set(self.y))
        )

        total_params = sum(p.numel() for
                           p in network.parameters() if p.requires_grad)

        # number of weights from input layer
        weights = self.x.shape[1] * 64
        # number of biases input layer
        weights += 64
        # number of weights between remaining layers
        weights += 64 * 64 * 4
        # number of weights between output layer and last layer
        weights += 64 * 2
        # number of biases - 4 layers + output layer
        weights += 64 * 4 + 2
        # batch norm for the first resblock
        # since it is applied to the input layer
        weights += 64 * 2
        # batch norm weights for the 3 layers
        weights += 64 * 3 * 2

        self.assertEqual(weights, total_params)

    def test_fcnet_all_reg(self):

        config = get_fixed_fcresnet_config(
            num_res_blocks=2,
            activate_dropout='Yes',
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

        # number of units from input layer to
        # first res block layer
        weights = 64 * 64
        # number of weights between remaining layers
        weights += 64 * 64 * 3
        # batch norm for the first resblock
        # since it is applied to the input layer
        weights += 64 * 2
        # batch norm weights for the 3 layers
        weights += 64 * 3 * 2
        # number of biases - 4 layers
        weights += 64 * 4
        # because of shake shake we have double
        # of the above
        weights *= 2
        # number of weights from input layer
        weights = self.x.shape[1] * 64
        # number of biases for input layer
        weights += 64
        # number of weights between output layer and last layer
        weights += 64 * 2
        # number of biases output layer
        weights += 2

        self.assertEqual(weights, total_params)


if __name__ == '__main__':
    unittest.main()
