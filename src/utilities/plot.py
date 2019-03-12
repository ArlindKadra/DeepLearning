import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
import pickle
import json
import math

from scipy.stats import spearmanr


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
                # append test set result for budget
                if budget not in accuracies:
                    accuracies[budget] = []
                if not math.isinf(test_loss):
                    accuracies[budget].append(test_loss)

                # create dict for each budget
                if budget not in val_losses:
                    val_losses[budget] = {}
                # for each budget add dicts with test loss as key and
                # val loss over epochs
                val_losses[budget][test_loss] = val_loss_epochs
            else:
                # Sometimes the worker dies unexpectedly
                # better to just pass here
                # raise ValueError("Empty Info field for the worker run")
                pass

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

    budgets = list(accuracies.keys())
    budgets.sort()
    ax.set_xticklabels(budgets)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel("Budget (epochs)")
    ax.set_ylabel("Test Loss")
    # plt.show()
    # Save the figure
    plt.savefig(os.path.join(working_dir, 'budget_test_loss.pdf'), bbox_inches='tight')
    plt.close(fig)


def best_conf_val_loss(working_dir):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)

    fig = plt.figure(2)  # an empty figure with no axes
    run_id = result.get_incumbent_id()
    runs = result.get_runs_by_id(run_id)
    best_run = runs[-1]
    info = best_run.info
    budget = int(info['nr_epochs'])
    epochs = [x for x in range(1, budget + 1)]
    validation_curve = info['val_loss']
    train_curve = info['train_loss']
    ax = plt.subplot(111)
    plt.plot(epochs, validation_curve, label='Validation', color='#f92418')
    plt.plot(epochs, train_curve, label='Train', color='#1b07fc')
    # Check if we have bounds and
    # if we do, get them.
    if 'train_loss_min' in info:
        min_train_curve = info['train_loss_min']
        max_train_curve = info['train_loss_max']
        min_val_curve = info['val_loss_min']
        max_val_curve = info['val_loss_max']
        plt.fill_between(epochs, min_train_curve, max_train_curve, color='#1b07fc', alpha=0.2)
        plt.fill_between(epochs, min_val_curve, max_val_curve, color='#f92418', alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Best Hyperparameter Configuration")
    ax.legend(loc='best')
    # plt.show()
    # Save the figure
    plt.savefig(os.path.join(working_dir, 'validation_curve.pdf'), bbox_inches='tight')
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
        plt.xlim(0, 240)

        for i in range(0, 4):
            plt.plot(ordered_val_curves[i], color='%s' % colors[color_index])

        color_index += 1
        plot_index += 1

    # plt.show()
    plt.savefig(os.path.join(working_dir, 'top_curves.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_rank_correlations(working_dir):

    with open(os.path.join(working_dir, "results.pkl"), "rb") as fp:
        result = pickle.load(fp)

    low_fid = []
    high_fid = []
    runs = result.get_learning_curves()
    fig = plt.figure(4)

    budgets = [30.0, 60.0, 120.0, 240.0]
    for i in range(0, len(budgets)):
        for j in range(i + 1, len(budgets)):
            small_fid = budgets[i]
            big_fidelity = budgets[j]
            # for each run a dict of config ids as keys and
            # a list of lists. The inner list contains tuples
            for conf_id in runs.keys():
                # get the inner list
                configs = runs[conf_id][0]
                if configs[0][0] == small_fid and configs[-1][0] == big_fidelity:
                    low_fid.append(configs[0][1])
                    high_fid.append(configs[-1][1])
            plt.scatter(low_fid, high_fid)
            plt.xlabel("%.0f" % small_fid)
            plt.ylabel("%.0f" % big_fidelity)
            rco, p = spearmanr(low_fid, high_fid)
            title = "Fidelity %.0f, %.0f \n" \
                    "Rank correlation %f (p value %f)" % (small_fid, big_fidelity, rco, p)
            plt.title(title)
            low_fid.clear()
            high_fid.clear()

            plt.savefig(
                os.path.join(
                    working_dir,
                    'fidelity%.0f%.0f.pdf' % (small_fid, big_fidelity)
                ),
                bbox_inches='tight'
            )
            plt.clf()
    plt.close(fig)
