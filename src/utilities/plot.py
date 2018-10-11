import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from scipy.stats import spearmanr
import numpy as np
import os
import pickle
import json

sns.set()


def load_data(working_dir):

    with open(os.path.join(working_dir, "results.json"), "r") as fp:

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


def test_loss_over_budgets(working_dir):
    
    accuracies, val_losses = load_data(working_dir)
    # an empty figure with no axes
    fig = plt.figure(1)
    # Create an axes instance
    ax = plt.subplot(111)
    data_plot = []
    for budget, values in accuracies.items():
        data_plot.append(values)
    plt.boxplot(data_plot)
    ax.set_xticklabels(['9', '27', '81', '243'])
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel("Budget (epochs)")
    ax.set_ylabel("Test Loss")
    ax.set_title("AutoFCResnet")
    # plt.show()
    # Save the figure
    plt.savefig(os.path.join(working_dir, 'budget_test_loss.png'), bbox_inches='tight')
    plt.close(fig)


def best_conf_val_loss(working_dir):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)

    fig = plt.figure(2)  # an empty figure with no axes
    run_id = result.get_incumbent_id()
    runs = result.get_runs_by_id(run_id)
    best_run = runs[-1]
    # budget = best_run.budget
    validation_curve = best_run.info['val_loss']
    train_curve = best_run.info['train_loss']
    ax = plt.subplot(111)
    plt.plot(validation_curve, label='Validation')
    plt.plot(train_curve, label='Train')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Best Hyperparameter Configuration")
    ax.legend(loc='best')
    # plt.show()
    # Save the figure
    plt.savefig(os.path.join(working_dir, 'validation_curve.png'), bbox_inches='tight')
    plt.close(fig)


def plot_curves(working_dir):

    fig = plt.figure(3)
    # colors to be used for plot
    colors = ['#fbff0f', '#fcc911', '#ed1b04', "#200cd6"]

    accuracies, val_losses = load_data(working_dir)
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
        plt.xlim(0, 242)

        for i in range(0, 4):
            plt.plot(ordered_val_curves[i], color='%s' % colors[color_index])

        color_index += 1
        plot_index += 1

    # plt.show()
    plt.savefig(os.path.join(working_dir, 'top_curves.png'), bbox_inches='tight')
    plt.close(fig)


def plot_rank_correlations(working_dir):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)

    values = OrderedDict()
    fig = plt.figure(4)  # an empty figure with no axes
    runs = result.get_learning_curves()

    # for each run a dict of config ids as keys and
    # a list of lists. The inner list contains tuples
    # with budget and loss
    for conf_id in runs.keys():
        # get the inner list
        configs = runs[conf_id][0]
        # this config was run for all budgets
        if len(configs) == 4:
            for config in configs:
                if config[0] not in values:
                    values[config[0]] = []
                values[config[0]].append(config[1])

    budgets = values.keys()
    rho, _ = spearmanr([losses for key, losses in values.items()], axis=1)
    mask = np.zeros_like(rho)
    # remove redudant information since the matrix is symmetrical
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(
            rho, mask=mask, vmax=1,
            annot=True, yticklabels=budgets,
            xticklabels=budgets, square=True,
            cmap="YlGnBu"
        )
        ax.set_title("Rank correlation")
        plt.savefig(os.path.join(
            "/home/kadraa/Documents/task_12/fcresnet",
            'rank_correlations.png'),
            bbox_inches='tight')
        plt.close(fig)