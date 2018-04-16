import time
import numpy as np
import random

from hpbandster.core.worker import Worker


class Slave(Worker):

    def compute(self, config, budget, *args, **kwargs):
        """
            Simple example for a compute function

            The loss is just a the config + some noise (that decreases with the budget)
            There is a 10 percent failure probability for any run, just to demonstrate
            the robustness of Hyperband agains these kinds of failures.

            For dramatization, the function sleeps for one second, which emphasizes
            the speed ups achievable with parallel workers.
        """
        # TODO

        res = []
        int(budget)
        tmp = network.train(config, 5, x_train, y_train, x_test, y_test)
        res.append(tmp)

        return ({
            'loss': np.mean(res),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })

