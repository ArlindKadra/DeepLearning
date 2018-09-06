import pickle
import matplotlib.pyplot as plt
import os
import json


def plot_best_config_val():
    with open("C:\\Users\\Lindarx\\Desktop\\small networks\\task_3\\fcresnet\\results.pkl", "rb") as fp:
        result = pickle.load(fp)
    # fig = plt.figure(2)  # an empty figure with no axes
    run_id = result.get_incumbent_id()
    runs = result.get_runs_by_id(run_id)
    best_run = runs[-1]
    # budget = best_run.budget
    validation_curve = best_run.info['val_loss']
    ax = plt.subplot(111)
    plt.plot(validation_curve)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    test_accuracy = best_run.info['test_accuracy']
    print("Test accuracy")
    config_info = result.get_id2config_mapping()
    print(str(test_accuracy))
    print("Config Info")
    print(str(config_info[run_id]))
    ax.set_title("Best Hyperparameter Configuration")
    plt.show()


def load_data():

    with open(os.path.join("C:\\Users\\Lindarx\\Desktop\\small networks\\task_3\\fcresnet", "results.json"), "r") as fp:

        accuracies = {}
        val_losses = {}
        for line in fp:

            output = json.loads(line)
            budget = output[1]
            result = output[3]

            if result is not None:
                _ = result['info']
                val_loss_epochs = _['val_loss']
                test_loss = _['test_loss']
            else:
                raise ValueError("Empty Info field for the worker run")

            # append test set result for budget
            if budget not in accuracies:
                accuracies[budget] = []
            accuracies[budget].append(test_loss)

            # create dict for each budget
            if budget not in val_losses:
                val_losses[budget] = {}
            # for each budget add dicts with test loss as key and
            # val loss over epochs
            val_losses[budget][test_loss] = val_loss_epochs

        return accuracies, val_losses


def plot_curves():

    # fig = plt.figure(3)
    # colors to be used for plot
    colors = ['#fbff0f', '#fcc911', '#ed1b04', "#200cd6"]

    accuracies, val_losses = load_data()
    # plt.rcParams.update({'axes.titlesize': 'large'})

    color_index = 0
    plot_index = 1

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for budget in val_losses:
        validation_curves = val_losses[budget]
        ordered_val_curves = [value for key, value in sorted(validation_curves.items())]
        ax = plt.subplot(2, 2, plot_index)
        if plot_index != 2 and plot_index != 1:
            ax.set_xlabel("Epoch")

        ax.set_ylabel("Validation Loss")
        ax.set_title("Top 5 configs for budget %d" % int(budget))

        for i in range(0, 4):
            plt.plot(ordered_val_curves[i], color='%s' % colors[color_index])

        color_index += 1
        plot_index += 1

    plt.show()


plot_best_config_val()
