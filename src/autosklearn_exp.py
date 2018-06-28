from model import Loader
from models import random_forest
from models import svm
import utils

import logging
import argparse


def main():

    parser = argparse.ArgumentParser(description='AutoSkLearn on OpenML.')
    parser.add_argument('--run_id', help='unique id to identify the AutoSklearn run.',
                        default='AutoSKLearn on OpenML.', type=str)

    args = parser.parse_args()

    # initialize logging
    logger = logging.getLogger(__name__)
    # TODO put verbose into configuration file
    verbose = True
    utils.setup_logging("AutoSklearn" + args.run_id, logging.DEBUG if verbose else logging.INFO)
    logger.info('AutoSkLearn Experiment started')

    loader = Loader(torch=False)
    x, y, categorical = loader.get_dataset()

    random_forest.train(x, y, categorical)
    #svm.train(x, y, categorical)


if __name__ == '__main__':
    main()