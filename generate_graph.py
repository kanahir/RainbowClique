import pandas as pd
import networkx as nx
import numpy as np
import time
from itertools import permutations, combinations
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import random
import pickle

MAX_COLORS=50
MIN_COLORS = 10

def create_graph(num):
    matrix = pd.read_csv("M_251189.CSV", header=None)
    matrix = matrix.drop([1094, 1095], axis=1)
    adj = matrix[1:]
    adj = adj[:-1]
    labels = matrix[:1].to_numpy()
    n_labels = len(set(labels[0, :]))
    if n < n_labels:
        return
    my_dict = {}
    for l in range(len(labels[0, :])):
        my_dict.update({l: int(labels[0, l])})
    g = nx.Graph()
    keys = list(my_dict.keys())
    for key in keys:
        g.add_node(key, label=my_dict[key])
    adj_np = adj.to_numpy()
    edges_dict = {k: [] for k in keys}
    for col in range(adj_np.shape[1]):
        b = adj_np[:, col]
        for i in range(b.shape[0]):
            if b[i] == 1:
                edges_dict[col].append(i)
    nodes = list(g.nodes())
    for n in nodes:
        for neigh in edges_dict[n]:
            g.add_edge(n, neigh)
    labels_clique = list(set(my_dict.values()))
    d = {j: [] for j in labels_clique}
    for vv in keys:
        d[my_dict[vv]].append(vv)
    lens = [len(h) for h in list(d.values())]
    avg_len = np.mean(lens)
    if num != 0:
        d, my_dict, g = add_class(g, avg_len, d, my_dict, num=num-n_labels)
    labels_clique = list(set(my_dict.values()))
    # print("there are {} labels".format(len(labels_clique)))
    # g = add_nodes_edges(g, my_dict)
    return g, my_dict, d, avg_len


def add_class(g, avg_len, d, my_dict, num=1):
    for it in range(num):
        nodes = list(g.nodes())
        new_class = max(list(d.keys())) + 1
        new_nodes = [j for j in range(max(nodes) + 1, int(avg_len) + max(nodes))]
        for n in new_nodes:
            nodes.append(n)
        d.update({new_class: new_nodes})
        my_dict.update({k: new_class for k in new_nodes})
        g = add_nodes_edges(g, my_dict)
    return d, my_dict, g


def add_nodes_edges(g, my_dict):
    old_g = g.copy()
    keys = list(my_dict.keys())
    new_nodes = list(set(keys) - set(g.nodes()))
    g.add_nodes_from(new_nodes)
    p1 = old_g.number_of_edges() / (old_g.number_of_nodes() * (old_g.number_of_nodes() - 1) / 2)
    miss = [pair for pair in combinations(g.nodes(), 2) if
            not g.has_edge(*pair) and my_dict[pair[0]] != my_dict[pair[1]]]
    old = [pair for pair in combinations(old_g.nodes(), 2) if
           not old_g.has_edge(*pair) and my_dict[pair[0]] != my_dict[pair[1]]]
    final_miss = list(set(miss) - set(old))
    num_edges = int(p1 * (g.number_of_nodes() * (g.number_of_nodes() - 1) / 2))
    current_num_edges = g.number_of_edges()
    random.shuffle(final_miss)
    g.add_edges_from(final_miss[:num_edges - current_num_edges])
    return g


if __name__ == '__main__':
    # Generating a multi-partite graph with MIN_COLORS classes and adding one class at a time until reaching MAX_COLORS
    file_name = "graph"
    graph_, node_to_label, label_to_node, avg_len_label = create_graph(num=MIN_COLORS)
    for i in range(MIN_COLORS-1, MAX_COLORS):
        d, node_to_label, graph_ = add_class(graph_, avg_len_label, label_to_node, node_to_label, num=1)
        temp_for_save = (graph_, node_to_label, label_to_node)
        nx.write_gpickle(temp_for_save, file_name + "_{}_classes".format(i))