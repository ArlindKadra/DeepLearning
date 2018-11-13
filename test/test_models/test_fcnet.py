import unittest

from src.models.fcnet import FcNet
import src.model as model
from src.utilities.search_space import get_fixed_fc_config


class TestFcNet(unittest.TestCase):

    def setUp(self):

        model.Loader(3)

        self.x, self.y, _ = model.get_dataset()


    def test_fcnet_no_reg(self):

        config = get_fixed_fc_config(max_nr_layers=3).sample_configuration()

        network = FcNet(
            config,
            self.x.shape[1],
            len(set(self.y))
        )

        total_params = sum(p.numel() for
                           p in network.parameters() if p.requires_grad)

        # number of weights from first layer to input
        weights = self.x.shape[1] * 64
        # number of weights between remaining layers
        weights += 64 * 64 * 2
        # number of weights between output layer and last layer
        weights += 64 * 2
        # number of biases - 3 layers + output layer
        weights += 64 * 3 + 2
        # batch norm weights
        weights += 64 * 3 * 2
        self.assertEqual(weights, total_params)




if __name__ == '__main__':
    unittest.main()