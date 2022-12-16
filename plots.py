import math
import matplotlib.pyplot as plt


def plot_histogram(data, x_label, y_label, bins):
    """
    Plot histogram.

    :param data: numbers in a list
    :param x_label: label on x axis
    :param y_label: label on y axis
    :param bins: number of bins in a histogram
    :return: None
    """
    plt.hist(data, density=False, bins=bins)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_multiple_histograms(data, num_tasks, metrics, title, colors, y_label, y_min):
    """
    Plot multiple vertical bars for each task.

    :param data: dictionary with mean and std data for each task
    :param num_tasks: number of tasks trained
    :param metrics: list of strings - possibilities: 'acc', 'auroc', 'auprc'
    :param title: plot title (string)
    :param colors: list of colors used for bars (len(colors)=len(metrics))
    :param y_label: label of axis y (string)
    :param y_min: minimum y value
    :return: None
    """
    metrics_names = {
        'acc': 'Accuracy',
        'auroc': 'AUROC',
        'auprc': 'AUPRC'
    }

    font = {'size': 20}
    plt.rc('font', **font)
    plt.grid(axis='y')

    bar_width = 0.5
    for i, m in enumerate(metrics):
        heights = [data[i][m] for i in range(num_tasks)]
        x_pos = [(n * len(metrics)) + (i * bar_width) for n in range(num_tasks)]
        plt.bar(x_pos, heights, width=bar_width, color=colors[i], edgecolor='black',
                yerr=[data[i]['std_' + m] for i in range(num_tasks)], capsize=7, label=metrics_names[m])

        # # plot numbers of mean (height) on every bar
        # for j, x in enumerate(x_pos):
        #     plt.text(x - 0.2, y_min + 1, round(heights[j], 1), {'size': 10})

    ax = plt.gca()
    ax.set_ylim([y_min, 100])

    plt.xticks([(i * len(metrics)) + (math.floor(len(metrics) / 2) * bar_width) for i in range(num_tasks)],
               ['Task 1' if i == 0 else 'Task 1-%d' % (i+1) for i in range(num_tasks)])

    plt.ylabel(y_label)
    # plt.title(title)
    plt.legend()
    plt.show()


def plot_multiple_results(num_tasks, num_epochs, first_average, means, stds, legend_lst, title, colors, x_label, y_label, vertical_lines_x, vl_min, vl_max, show_CI=True, text_strings=None):
    """
    Plot more lines from the saved results on the same plot with additional information.

    :param num_tasks: number of tasks trained
    :param num_epochs: number of epochs per task
    :param first_average: string - show results on 'first' task only or the 'average' results until current task index
    :param means: list - [mean_acc, mean_auroc, mean_auprc]
    :param stds: list - [std_acc, std_auroc, std_auprc]
    :param legend_lst: list of label values (len(legend_lst)=len(means))
    :param title: plot title (string)
    :param colors: list of colors used for lines (len(colors)=len(means))
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :param vertical_lines_x: x values of where to draw vertical lines
    :param vl_min: vertical lines minimum y value
    :param vl_max: vertical lines maximum y value
    :param show_CI: show confidence interval range (boolean)
    :param text_strings: optional list of text strings to add to the bottom of vertical lines
    :return: None
    """
    font = {'size': 18}
    plt.rc('font', **font)
    plt.grid(axis='y')

    if num_tasks * num_epochs == len(means[0]):
        # plot horizontal lines to explain learning
        for i in range(num_tasks):
            x_min = i * num_epochs if i == 0 else (i * num_epochs) - 1
            x_max = ((i + 1) * num_epochs) - 1
            plt.hlines(y=103, xmin=x_min, xmax=x_max)
            plt.vlines(x=x_min, ymin=102, ymax=104)
            plt.vlines(x=x_max, ymin=102, ymax=104)

            plt.text(x=x_min + 1, y=104, fontsize=12,
                     s='Learning %s; Results for %s' % (str(i+1), str(1) if first_average == 'first' else '1-%s' % (i+1) if i != 0 else str(1)))

    # plot lines with confidence intervals
    i = 0
    for mean, std in zip(means, stds):
        # plot the shaded range of the confidence intervals (mean +/- std)
        if show_CI:
            up_limit = mean + std
            up_limit[up_limit > 100] = 100  # cut accuracies above 100
            down_limit = mean - std
            plt.fill_between(range(len(std)), up_limit, down_limit, color=colors[i], alpha=0.25)

        # plot the mean on top
        plt.plot(range(len(mean)), mean, colors[i], linewidth=3)

        i += 1

    if legend_lst:
        plt.legend(legend_lst, loc='lower left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.vlines(vertical_lines_x, vl_min, vl_max, colors='k', linestyles='dashed', linewidth=2, alpha=0.5)
    if text_strings is not None:
        for i in range(len(text_strings)):
            plt.text(vertical_lines_x[i] + 0.5, vl_min, text_strings[i], color='k', alpha=0.5)
    plt.show()


def plot_task_results(data, method_names, markers, colors, x_ticks, x_label, y_label):
    """
    Plot results for different methods.

    :param data: 2D list of results with the shape (num_methods, num_tasks)
    :param method_names: list of methods' names (len=num_methods)
    :param markers: list of point markers (len=num_methods)
    :param colors: list of colors (len=num_methods)
    :param x_ticks: list of strings to show at x axis (len=num_tasks)
    :param x_label: label of axis x (string)
    :param y_label: label of axis y (string)
    :return: None
    """
    font = {'size': 25}
    plt.rc('font', **font)
    plt.grid(axis='y')

    # plt.yticks([75, 80, 85, 90, 95])

    x = list(range(1, len(x_ticks) + 1))

    for i, method_res in enumerate(data):
        plt.plot(x, method_res, label=method_names[i], linestyle='--', color=colors[i], marker=markers[i],
                 linewidth=2, markersize=12)

    plt.xticks(x, x_ticks, fontsize=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_heatmap(data, full_value):
    """
    Plot heatmap for the ablation study.

    :param data: 2D list of shape (num_layers, num_layers)
    :param full_value: value of the result without ablation
    :return: None
    """
    import seaborn as sns
    import numpy as np

    font = {'size': 25}
    plt.rc('font', **font)

    data = np.array(data)
    data[data == 0] = full_value

    relative_diff = (-1 + (data / full_value)) * 100

    ax = sns.heatmap(relative_diff, cmap=sns.color_palette("Blues_r", as_cmap=True), linecolor='black', linewidth=1, square=True)

    for i in range(len(data)):
        for j in range(len(data[i])):
            if relative_diff[i, j] < 0:   # upper triangle
                text = ax.text(j + 0.5, i + 0.5, round(relative_diff[i, j], 1), ha="center", va="center", color="black")

    ax.set_xticklabels(list(range(1, 7)))
    ax.set_yticklabels(list(range(1, 7)))

    plt.xlabel('last ablated layer')
    plt.ylabel('first ablated layer')
    plt.show()

