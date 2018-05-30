import logging
from utils import cross_validation
from models.fcresnet import FcResNet
from models import fcresnet

from hpbandster.api.optimizers.bohb import BOHB
import hpbandster.api.util as hputil
from .hpbandster import Slave

import numpy as np
import os
import argparse
import pickle

from hpbandster.core.worker import Worker

class Master():

    def __init__(self, run_id, array_id, working_dir):

        config_space = fcresnet.get_config_space()


        if array_id == 1:
            # start nameserver
            NS = hputil.NameServer(run_id=run_id, nic_name='eth0',
                                   working_directory=working_dir)

            # stores information for workers to find in working_directory
            ns_host, ns_port = NS.start()

            # BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
            worker = Slave(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
            worker.run(background=True)

            HPB = BOHB(configspace=config_space,
                       run_id=run_id,
                       eta=3, min_budget=1, max_budget=27,
                       host=ns_host,
                       nameserver=ns_host,
                       nameserver_port=ns_port,
                       ping_interval=3600,
                       )
            # BOHB can wait until a minimum number of workers is online before starting
            res = HPB.run(n_iterations=4,
                          min_n_workers=4
                          )

            # pickle result here for later analysis
            with open(os.path.join(working_dir, 'results.pkl'), 'wb') as fh:
                pickle.dump(res, fh)

            # shutdown all workers
            HPB.shutdown(shutdown_workers=True)

            # and the nameserver
            NS.shutdown()

        else:

            host = hputil.nic_name_to_host('eth0')

            # workers only instantiate the MyWorker, find the nameserver and start serving
            w = Slave(run_id=run_id, host=host)
            w.load_nameserver_credentials(working_dir)
            # run worker in the foreground,
            w.run(background=False)


class Slave(Worker):

    def compute(self, config, budget, *args, **kwargs):
        """All the functionality that the worker will compute.

        The worker will train the neural network and
        save results related to the performance.

        Args:
            config: A hyperparameter configuration drawn from the ConfigSpace.
            budget: budget on which the training of the network will be limited.
            kwargs: The input and labels of the network
            should be found in this dictionary.
        """

        if "input" not in kwargs:
            raise KeyError("No input given for the network")
        if "labels" not in kwargs:
            raise KeyError("No labels given for the network")

        loss, accuracy = cross_validation(int(budget), kwargs["input"], kwargs["labels"], FcResNet, config)
        return ({
            'loss': np.mean(loss),  # this is the a mandatory field to run hyperband
            'info': (loss, accuracy)  # can be used for any user-defined information - also mandatory
        })
