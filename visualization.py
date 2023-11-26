import numpy as np
import matplotlib.pyplot as plt
import math


def plot_clique_size(clique_size_list, titles_list):
    plt.figure(0, figsize=(9, 7))
    classes = [i + 9 for i in range(len(clique_size_list[0]))]
    plt.xticks(np.arange(min(classes), max(classes) + 1, 1))
    # colors = ['blue', 'orange', 'g']
    colors = ['m']
    for clique_size, title, color in zip(clique_size_list, titles_list, colors):
        plt.plot(classes, clique_size, linewidth=3, label=title, color=color)
    plt.title("Max clique size VS Number of Classes")
    plt.xlabel("Number of classes")
    plt.ylabel("Max clique size")
    plt.legend(fontsize='xx-large')
    plt.grid()
    plt.savefig("max_clique_size.png")
    plt.show()


def plot_all_times(times_lists, titles_list):
    plt.figure(0, figsize=(9, 7))
    classes = [i + 9 for i in range(len(times_lists[0]))]
    log_class = [math.log10(x) for x in classes]
    # plt.xticks(np.arange(min(log_class), max(log_class) + 1, 1))
    colors = ['orange', 'g', 'r', 'm']
    # colors = ['m']
    for times_list, title, color in zip(times_lists, titles_list, colors):
        log_time = [math.log10(x) for x in times_list]
        plt.plot(log_class, log_time, linewidth=3, label=title, color=color)
    plt.title("Running Time VS Number of Classes")
    plt.xlabel("Number of classes (log 10)")
    plt.ylabel("Running time [seconds] (log 10)")
    plt.legend(fontsize='xx-large')
    plt.grid()
    plt.savefig("running_time.png")
    plt.show()


def plot_lonely_nodes(n_lonely_nodes):
    plt.clf()
    plt.plot(range(len(n_lonely_nodes)), n_lonely_nodes, color="black", linewidth=3)
    plt.title("Number of Lonely Nodes VS Number of Classes")
    plt.savefig("lonely_nodes.png")
    plt.show()


if __name__ == '__main__':
    pass
