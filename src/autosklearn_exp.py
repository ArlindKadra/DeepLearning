import model
from models import random_forest
from models import svm
from models import gradient_boosting
from utilities.log import setup_logging

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
        default=1,
        type=int,
    )
    parser.add_argument(
        '--task_id',
        help='Task id so that the dataset can be retrieved from OpenML.',
        default=3,
        type=int
    )

    args = parser.parse_args()
    algorithms = {1: random_forest, 2: svm, 3: gradient_boosting}
    network = algorithms[args.array_id]
    # TODO put verbose into configuration file
    verbose = True
    setup_logging("AutoSklearn" + str(args.run_id) + "_" + str(args.array_id),
                  logging.DEBUG if verbose else logging.INFO)
    model.Loader(task_id=args.task_id, torch=False)
    x, y, categorical = model.get_dataset()
    network.train(x, y, categorical, args.task_id)


if __name__ == '__main__':
    main()
