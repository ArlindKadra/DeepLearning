from models.random_forest import RandomForest
from models.svm import SVM
from models.gradient_boosting import GradientBoosting
from utilities.log import setup_logging
import openml

import logging
import argparse
import os
import re


def get_data(task_id):

    logger = logging.getLogger(__name__)
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    x, y, categorical = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True
    )

    logger.info("Data from task id %d Loaded", task_id)

    # TODO use mean or other ways to fill the missing values.
    # For the moment, do not deal with input that contains NaN values
    return x, y, categorical


def get_split_indices(task_id):

    task = openml.tasks.get_task(task_id)
    train_indices, test_indices = task.get_train_test_split_indices()

    return (train_indices, test_indices)


def id_to_algorithm_name(id):

    if id == 1:
        return 'Random Forest'
    elif id == 2:
        return 'SVM'
    else:
        return 'Gradient Boosting'


def main():

    time_tasks = {"233": 488.28747844696045, "236": 8158.206694602966, "241": 687.9558777809143,
                  "242": 3554.382823228836, "244": 1934.4902665615082, "246": 2160.3169343471527,
                  "248": 2540.9990570545197, "252": 1561.785254240036, "253": 1659.6252002716064,
                  "261": 612.370374917984, "262": 12782.797701835632, "267": 39.622684478759766,
                  "273": 971.8062009811401, "275": 3787.1223397254944, "279": 237.6554594039917,
                  "283": 166.36031579971313, "336": 20847.72578406334, "2120": 6144.490860700607,
                  "3047": 379.1345524787903, "75169": 8747.143706083298, "146577": 131.11559891700745,
                  "75092": 1819.514637708664, "75129": 940.6834437847137, "146583": 214.23687982559204,
                  "75099": 1437.2833738327026, "75159": 254.31494808197021, "146596": 322.1039328575134,
                  "75227": 5763.138130664825, "75232": 1731.6893405914307, "75235": 7447.60431599617,
                  "75236": 1155.4758067131042, "146593": 363.27844882011414, "146594": 4785.427848577499,
                  "126026": 43941.4469435215, "75225": 3481.9932889938354, "75221": 664.8377544879913,
                  "146586": 35.47628879547119, "146587": 470.55258798599243, "75215": 1900.301148891449,
                  "146585": 129951.86498236656, "75109": 15302.564419984818, "146591": 4849.020035743713,
                  "146604": 1770.3917262554169, "146680": 69081.15261077881, "168785": 1329.64701795578,
                  "167106": 328.78990387916565, "167105": 5208.157749414444, "168786": 825.4804713726044,
                  "168787": 1942.231296300888, "168788": 4249.048784255981, "168789": 124559.49959087372,
                  "168790": 74805.81290125847, "167083": 179456.2090280056, "167204": 236823.751642704,
                  "168791": 8005.748018026352, "167202": 3638.0211086273193, "167097": 3239.2690336704254,
                  "168792": 112296.3499071598, "168794": 36009.80317401886, "168795": 69165.76284074783,
                  "168798": 25694.68651843071}

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
        default=3,
        type=int,
    )
    parser.add_argument(
        '--task_id',
        help='Task id so that the dataset can be retrieved from OpenML.',
        default=3,
        type=int
    )

    args = parser.parse_args()
    algorithms = {1: RandomForest, 2: SVM, 3: GradientBoosting}
    algorithm = algorithms[args.array_id]
    # TODO put verbose into configuration file
    verbose = True
    run_id = re.sub(r"\D+\d+(\d|\])*$", "", args.run_id)
    setup_logging("AutoSklearn" + str(run_id) + "_" + str(args.array_id),
                  logging.DEBUG if verbose else logging.INFO)

    x, y, categorical = get_data(args.task_id)
    split_indices = get_split_indices(args.task_id)
    time = int(time_tasks['%d' % args.task_id])
    method = algorithm(x, y, categorical, split_indices, args.task_id)
    method.train(time, run_id)
    accuracy = method.predict()

    path = os.path.expanduser(
        os.path.join(
            '~',
            'AutoSklearn',
            id_to_algorithm_name(args.array_id),
            str(args.task_id)
        )
    )

    if os.path.exists(path):
        if not os.path.isdir(path):
            os.makedirs(path)
    else:
        os.makedirs(path)

    with open(os.path.join(path, 'performance.txt'), 'a') as fp:
        fp.write('%f\n' % accuracy)


if __name__ == '__main__':
    main()

