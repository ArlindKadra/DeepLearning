import model
from models import random_forest
from models import svm
import utils

import logging
import argparse


def main():

    parser = argparse.ArgumentParser(description='AutoSkLearn on OpenML.')
    parser.add_argument(
        '--run_id',
        help='unique id to identify the AutoSklearn run.',
        default='AutoSKLearn on OpenML.',
        type=str
    )
    parser.add_argument(
        '--array_id',
        help='SGE array id.',
        default=1
        type=int
    )
    parser.add_argument(
        '--task_id',
        help='Task id so that the dataset can be retrieved from OpenML.',
        default=3,
        type=int
    )

    args = parser.parse_args()
    algorithms = {1:random_forest, 2:svm}
    network = algorithms[args.array_id]
    # initialize logging
    logger = logging.getLogger(__name__)
    # TODO put verbose into configuration file
    verbose = True
    utils.setup_logging("AutoSklearn" + args.run_id, logging.DEBUG if verbose else logging.INFO)
    logger.info('AutoSkLearn Experiment started')
    model.Loader(task_id=args.task_id, torch=False)
    x, y, categorical = model.get_dataset()

    network.train(x, y, categorical, args.task_id)


if __name__ == '__main__':
    main()
