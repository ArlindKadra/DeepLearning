from utilities import plot, log
from optim.hpbandster import Master
import utils
import model
from model import Loader
import os
import argparse
import logging


def main():

    parser = argparse.ArgumentParser(description='Fully connected residual network')
    parser.add_argument('--num_workers', help='Number of hyperband workers.', default=4, type=int)
    parser.add_argument('--num_iterations', help='Number of hyperband iterations.', default=4, type=int)
    parser.add_argument('--run_id', help='unique id to identify the HPB run.', default='HPB_example_2', type=str)
    parser.add_argument('--array_id', help='SGE array id to tread one job array as a HPB run.', default=1, type=int)
    parser.add_argument('--working_dir', help='working directory to store live data.', default='.', type=str)
    parser.add_argument('--nic_name', help='name of the Network Interface Card.', default='lo', type=str)
    parser.add_argument('--task_id', help='Task id so that the dataset can be retrieved from OpenML.',
                        default=3, type=int)

    args = parser.parse_args()

    # initialize logging
    logger = logging.getLogger(__name__)
    # TODO put verbose into configuration file
    verbose = False
    utils.setup_logging(args.run_id + "_" + str(args.array_id), logging.DEBUG if verbose else logging.INFO)
    logger.info('DeepResNet Experiment started')
    Loader(args.task_id)
    working_dir = os.path.join(args.working_dir, 'task_%i' % model.get_task_id(), 'fcresnet')
    Master(args.num_workers, args.num_iterations, args.run_id, args.array_id, working_dir, args.nic_name)
    # config_space = fcresnet.get_config_space()
    # config = config_space.sample_configuration(1)
    # result = utils.cross_validation(x, y, fcresnet.FcResNet, config)
    # logging.info('Cross Validation loss: %.3f, accuracy %.3f', result[0], result[1])
    plot.test_loss_over_budgets(working_dir)
    plot.best_conf_val_loss(working_dir)
    plot.plot_curves(working_dir)
    log.save_best_config(working_dir)


if __name__ == '__main__':
    main()
