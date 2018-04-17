import logging
from src.utils import cross_validation
from src.models.fcresnet import FcResNet
from src.models import fcresnet

from hpbandster.api.optimizers.bohb import BOHB
import hpbandster.api.util as hputil
from .hpbandster import Slave

import numpy as np
import os
import argparse
import pickle

from hpbandster.core.worker import Worker

class Master():

    config_space = fcresnet.get_config_space()
    parser = argparse.ArgumentParser(description='HpBandSter example 2.')
    parser.add_argument('--run_id', help='unique id to identify the HPB run.', default='HPB_example_2', type=str)
    parser.add_argument('--array_id', help='SGE array id to tread one job array as a HPB run.', default=1, type=int)
    parser.add_argument('--working_dir', help='working directory to store live data.', default='.', type=str)

    args = parser.parse_args()

    if args.array_id == 1:
        # start nameserver
        NS = hputil.NameServer(run_id=args.run_id, nic_name='eth0',
                               working_directory=args.working_dir)

        # stores information for workers to find in working_directory
        ns_host, ns_port = NS.start()

        # BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
        worker = Slave(nameserver=ns_host, nameserver_port=ns_port, run_id=args.run_id)
        worker.run(background=True)

        HPB = BOHB(configspace=config_space,
                   run_id=args.run_id,
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
        with open(os.path.join(args.working_dir, 'results.pkl'), 'wb') as fh:
            pickle.dump(res, fh)

        # shutdown all workers
        HPB.shutdown(shutdown_workers=True)

        # and the nameserver
        NS.shutdown()

    else:

        host = hputil.nic_name_to_host('eth0')

        # workers only instantiate the MyWorker, find the nameserver and start serving
        w = Slave(run_id=args.run_id, host=host)
        w.load_nameserver_credentials(args.working_dir)
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
