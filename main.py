
import networkx as nx
import numpy as np
import time
from itertools import combinations
import matplotlib as mpl
import visualization
from operator import itemgetter

N_ITERATION = 10


mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 15

NAX_COLORS = 66



def greedy(clique, g, nodes, label_to_node_, node_to_label_, trip=False):
    """
    The algorithm use greedy method to find max clique in graph (add the next node with biggest rank).
    In order to find other cliques it gets 3 nodes in the previous clique (if it isn't maximal) and build clique from them
    :param clique: the clique until now
    :param g: a graph
    :param nodes: a list of the nodes in the graph
    :param label_to_node_: a dictionary of label and the nodes in this label
    :param node_to_label_: a dictionary of node and it's label
    :param trip: True if the algorithm start with a triplet, False otherwise.
    :return: Thw maximal clique it founded
    """
    if trip:
        for v in clique:
            not_connected = [n for n in nodes if not g.has_edge(n, v) and n != v]
            g.remove_nodes_from(not_connected)
            nodes = list(g.nodes())
        node2deg = {n: g.degree(n) for n in nodes if n not in clique}
        node_max_deg = max(node2deg, key=node2deg.get)
        clique.append(node_max_deg)
    while len(clique) != len(label_to_node_):
        v = clique[-1]
        not_connected = [n for n in nodes if not g.has_edge(n, v) and n != v]
        g.remove_nodes_from(not_connected)
        nodes = list(g.nodes())
        node2deg = {n: g.degree(n) for n in nodes if n not in clique}
        if node2deg == {}:
            return clique
        node_max_deg = max(node2deg, key=node2deg.get)
        clique.append(node_max_deg)
    return clique


def bron_kerbosch(graph, labels, label_to_node_, potential_clique, remaining_nodes, skip_nodes, found_cliques=[]):
    """
    The algorithm using bron-kerbosch algorithm in order to find cliques in graph.
    The algorithm stops if it found clique with all labels (max potential clique)
    :param graph: a graph
    :param labels: a list off all labels in the graph
    :param label_to_node_: a dictionary of label and the nodes in this label
    :param potential_clique: the builded clique until now
    :param remaining_nodes: the potential nodes to be in the clique
    :param skip_nodes: the nodes we already checked and found all cliques with them
    :param found_cliques: the cliques we found until now
    :return: all founded cliques in graph
    """
    if len(remaining_nodes) == 0 and len(skip_nodes) == 0:
        found_cliques.append(potential_clique)
        return found_cliques
    if len(potential_clique) == len(label_to_node_):
        found_cliques.append(potential_clique)
        return found_cliques

    for node in remaining_nodes:
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_skip_list = [n for n in skip_nodes if n in list(graph.neighbors(node))]
        found_cliques = bron_kerbosch(graph, labels, label_to_node_, new_potential_clique, new_remaining_nodes,
                                       new_skip_list, found_cliques)

        if len(found_cliques[-1]) == len(label_to_node_):
            return found_cliques

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes and add it to the skip list.
        remaining_nodes.remove(node)
        skip_nodes.append(node)

    return found_cliques


def color_greedy_ver1(graph, node_to_label, label_to_node_, labels_list, potential_clique, remaining_nodes, label_ind=0):
    """
     The algorithm rank the labels and build clique with this order (take less communicated labels first)
     :param graph: a graph
     :param nodes_to_label: a dictionary of node and it's label
     :param labels_to_node: a dictionary of label and the nodes in this label
     :param labels_list: a list of all labels in graph
     :param potential_clique: the builded clique until now
     :param remaining_nodes: the potential nodes to be in the clique
     :param label_ind: which label we add now
     :return: a clique that contain all labels
     """

    # check if success
    if len(potential_clique) == len(labels_list):
        return potential_clique

    # run over the nodes in this label with their rank
    potential_nodes_in_label = [n for n in label_to_node_[labels_list[label_ind]] if n in remaining_nodes]

    for node in potential_nodes_in_label:
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        clique_founded = color_greedy_ver1(graph, node_to_label, label_to_node_, labels_list, new_potential_clique,
                                         new_remaining_nodes,
                                         label_ind + 1)
        if clique_founded:
            return clique_founded

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes.
        remaining_nodes.remove(node)
    # if we failed
    return potential_clique


def color_greedy_ver2(graph, node_to_label, label_to_node_, nodes_dict, labels_list, potential_clique, remaining_nodes, added_labels):
    """
     The algorithm rank the labels and build clique with this order (take less communicate labels first)
     :param graph: a graph
     :param nodes_to_label: a dictionary of node and it's label
     :param labels_to_node: a dictionary of label and the nodes in this label
     :param labels_list: a list of all labels in graph
     :param potential_clique: the builded clique until now
     :param remaining_nodes: the potential nodes to be in the clique
     :param label_ind: which label we add now
     :return: a clique that contain all labels
     """
    # check if success
    if len(potential_clique) == len(labels_list):
        return potential_clique
    # check which label will be added next
    min_label = find_next_label(nodes_dict, added_labels, labels_list, node_to_label)
    # run of the nodes in this label with their rank
    potential_nodes_in_label = [n for n in label_to_node_[min_label] if n in remaining_nodes]
    while potential_nodes_in_label:
        # take max node and remove from potential
        node = find_max_node(potential_nodes_in_label, nodes_dict, added_labels)
        potential_nodes_in_label.remove(node)
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_added_labels = added_labels.copy()
        new_added_labels.append(min_label)
        clique_founded = color_greedy_ver2(graph, node_to_label, label_to_node_,nodes_dict, labels_list, new_potential_clique, new_remaining_nodes,
                                       new_added_labels)
        if clique_founded:
            return clique_founded

        # We're done considering this node.  If there was a way to form a clique with it, we
        # already discovered its maximal clique in the recursive call above.  So, go ahead
        # and remove it from the list of remaining nodes.
        remaining_nodes.remove(node)
    return potential_clique




def color_greedy_ver3(graph, node_to_label, label_to_node_, labels_list, potential_clique, remaining_nodes, added_labels, previous_cliques = []):

    """
     The algorithm rank the labels and build clique with this order (take less communicate labels first)
     :param graph: a graph
     :param nodes_to_label: a dictionary of node and it's label
     :param labels_to_node: a dictionary of label and the nodes in this label
     :param labels_list: a list of all labels in graph
     :param potential_clique: the builded clique until now
     :param remaining_nodes: the potential nodes to be in the clique
     :param label_ind: which label we add now
     :return: a clique that contain all labels
     """
    # check if success
    if len(potential_clique) == len(labels_list):
        return potential_clique
    # check which label will be added next
    min_label = find_next_label3(graph, label_to_node_, added_labels, labels_list)
    # run of the nodes in this label with their rank
    potential_nodes_in_label = [n for n in label_to_node_[min_label] if n in remaining_nodes]
    nodes_rank = [val for (node, val) in graph.degree(potential_nodes_in_label)]
    while potential_nodes_in_label:
        # take max node and remove from potential
        node = potential_nodes_in_label[np.argmax(nodes_rank)]
        potential_nodes_in_label.remove(node)
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_added_labels = added_labels.copy()
        new_added_labels.append(min_label)
        clique_founded = color_greedy_ver3(graph, node_to_label, label_to_node_, labels_list, new_potential_clique, new_remaining_nodes, new_added_labels)
        if clique_founded:
            return clique_founded
        remaining_nodes.remove(node)
    # return to next node in previous label
    return potential_clique


def color_greedy_ver4(graph, node_to_label, label_to_node_, labels_list, potential_clique, remaining_nodes, added_labels, cliques_founded = []):

    """
     The algorithm rank the labels and build clique with this order (take less communicate labels first)
     :param graph: a graph
     :param nodes_to_label: a dictionary of node and it's label
     :param labels_to_node: a dictionary of label and the nodes in this label
     :param labels_list: a list of all labels in graph
     :param potential_clique: the builded clique until now
     :param remaining_nodes: the potential nodes to be in the clique
     :param label_ind: which label we add now
     :return: a clique that contain all labels
     """
    # check if success
    if len(potential_clique) == len(labels_list) or len(remaining_nodes) == 0:
        return potential_clique
    # check which label will be added next
    min_label = find_next_label3(graph, label_to_node_, added_labels, labels_list)
    # run of the nodes in this label with their rank
    potential_nodes_in_label = [n for n in label_to_node_[min_label] if n in remaining_nodes]
    nodes_rank = [val for (node, val) in graph.degree(potential_nodes_in_label)]
    nodes_to_try = potential_nodes_in_label.copy()
    for _ in range(len(potential_nodes_in_label)):
        if len(nodes_to_try) == 0:
            break
        # take max node and remove from potential
        max_node_ind = np.argmax(nodes_rank)
        node = nodes_to_try[max_node_ind]
        nodes_rank.remove(nodes_rank[max_node_ind])
        nodes_to_try.remove(node)
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_added_labels = added_labels.copy()
        new_added_labels.append(min_label)
        cliques_founded.append(color_greedy_ver4(graph, node_to_label, label_to_node_, labels_list, new_potential_clique, new_remaining_nodes, new_added_labels, cliques_founded))
        if len(cliques_founded[-1]) == len(labels_list):
            return cliques_founded[-1]
        remaining_nodes.remove(node)
    if len(cliques_founded) == 0:
        return potential_clique
    else:
        return max(cliques_founded, key=lambda x: len(x))


def color_greedy_ver5(graph, node_to_label, label_to_node_, labels_list, potential_clique, remaining_nodes, added_labels, max_cliques_founded = []):

    """
     The algorithm rank the labels and build clique with this order (take less communicate labels first)
     :param graph: a graph
     :param nodes_to_label: a dictionary of node and it's label
     :param labels_to_node: a dictionary of label and the nodes in this label
     :param labels_list: a list of all labels in graph
     :param potential_clique: the builded clique until now
     :param remaining_nodes: the potential nodes to be in the clique
     :param label_ind: which label we add now
     :return: a clique that contain all labels
     """
    # check if success
    if len(potential_clique) == len(labels_list) or len(remaining_nodes) == 0:
        return potential_clique
    #check if found better clique than could be found
    future_possible_labels = [node_to_label[n] for n in remaining_nodes]
    if len(set(future_possible_labels)) + len(added_labels) <= len(max_cliques_founded):
        return max_cliques_founded
    # check which label will be added next
    min_label = find_next_label3(graph, label_to_node_, added_labels, labels_list)
    # run of the nodes in this label with their rank
    potential_nodes_in_label = [n for n in label_to_node_[min_label] if n in remaining_nodes]
    nodes_rank = [val for (node, val) in graph.degree(potential_nodes_in_label)]
    nodes_to_try = potential_nodes_in_label.copy()
    for _ in range(len(potential_nodes_in_label)):
        if len(nodes_to_try) == 0:
            break
        # take max node and remove from potential
        max_node_ind = np.argmax(nodes_rank)
        node = nodes_to_try[max_node_ind]
        nodes_rank.remove(nodes_rank[max_node_ind])
        nodes_to_try.remove(node)
        # Try adding the node to the current potential_clique to see if we can make it work.
        new_potential_clique = potential_clique + [node]
        new_remaining_nodes = [n for n in remaining_nodes if n in list(graph.neighbors(node))]
        new_added_labels = added_labels.copy()
        new_added_labels.append(min_label)
        cliques_founded = color_greedy_ver5(graph, node_to_label, label_to_node_, labels_list, new_potential_clique, new_remaining_nodes, new_added_labels, max_cliques_founded)
        if len(cliques_founded) == len(labels_list):
            return cliques_founded
        elif len(cliques_founded) > len(max_cliques_founded):
            max_cliques_founded = cliques_founded
        remaining_nodes.remove(node)
    return max_cliques_founded

def find_next_label3(graph, labels_to_node, added_labels, labels_list):
    average_rank = {label: np.mean(list(graph.degree(labels_to_node[label]))) for label in labels_list if label not in added_labels}
    return min(average_rank, key=average_rank.get)


def find_next_label(nodes_dict, added_labels, labels_list, nodes_to_label):
    """
    The function get a dictionary of nodes and the labels that added before and return the
     label with the fewest neighbors
    :param nodes_dict: a dictionary of nodes and the neighbors it has from each label
    :param added_labels: the label that we added before
    :param labels_list: a list of the labels
    :param nodes_to_label: a dictionary of node and it's label
    :return: the next label to add
    """
    # for each label find in how many nodes it is minimal
    minimal_number = {label: 0 for label in labels_list if label not in added_labels}
    for node, labels_dict in nodes_dict.items():
        # take relevant labels
        relavant_labels_dict = {key: labels_dict[key] for key in labels_list if key
                                not in added_labels and key != nodes_to_label[node]}
        if relavant_labels_dict != {}:
            minimal_label = min(relavant_labels_dict, key=relavant_labels_dict.get)
            minimal_number[minimal_label] += 1
    return max(minimal_number, key=minimal_number.get)


def find_max_node(nodes, nodes_dict, added_labels):
    """
    The function get nodes and return the next node to add to clique
    :param nodes: a list of nodes
    :param nodes_dict: a dictionary of nodes and the neighbors it has from each label
    :param added_labels: the non-relevant labels
    :return: the next node to choose
    """
    # take the sum of all neighbors
    n_neighbors = {}
    for node in nodes:
        node_neighbors = nodes_dict[node]
        relevant_neighbors = [node_neighbors[label] for label in node_neighbors if label not in added_labels]
        n_neighbors[node] = sum(relevant_neighbors)
    return max(n_neighbors, key=n_neighbors.get)


def rank_nodes_ver2(graph, label_to_node, nodes_to_label):
    """
    The function gives score for each node - the score is the minimum number of neighborhoods that it has from certain label.
    :param grapha: a graph
    :param label_to_node: dictionary of labels and all nodes with this label
    :param nodes_to_label: dictionary oo nodes and the label of each node
    :return: a dictionary of nodes, each node's value is a dictionary that contaain label
     and how many neighbors with this label the node has.
    """
    nodes_dict = {}
    for node in graph.nodes:
        connected_nodes = list(graph.neighbors(node))
        labels_dict = {label: 0 for label in label_to_node}
        for connected_node in connected_nodes:
            labels_dict[nodes_to_label[connected_node]] += 1
        nodes_dict[node] = labels_dict
    return nodes_dict


def rank_nodes_ver1(graph, label_to_node, nodes_to_label):
    """
    The function gives score for each node - the score is the minimum number of neighborhoods that it has from certain label.
    And give each label a score that depends how much it connect to other labels.
    :param grapha: a graph
    :param label_to_node: dictionary of labels and all nodes with this label
    :param nodes_to_label: dictionary oo nodes and the label of each node
    :return: a dictionary of nodes with their score and labels with their score.
    """
    n_labels = max(label_to_node)
    nodes_score = {}
    labels_score = {}
    for node in graph.nodes:
        connected_labels = np.zeros(n_labels + 1, dtype=int)
        connected_nodes = list(graph.neighbors(node))
        for connected_node in connected_nodes:
            connected_labels[nodes_to_label[connected_node]] += 1
        nodes_score[node] = min(connected_labels[np.nonzero(connected_labels)])
    for label in label_to_node:
        labels_score[label] = np.mean(itemgetter(*label_to_node[label])(nodes_score))

    return nodes_score, labels_score


def remove_lonely_nodes(graph, label_to_node, nodes_to_label):
    """
    The function get a graph and labels and remove the nodes that hasn't neighbor of each label
    :param graph: a graph
    :param label_to_node: dictionary of lables and all nodes with this label
    :return: the graph without all those nodes
    """
    lonely_nodes = []
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        for label in label_to_node:
            if nodes_to_label[node] != label and not len(set(neighbors).intersection(label_to_node[label])) !=0:
                lonely_nodes.append(node)
                del nodes_to_label[node]
                label_to_node[label].remove(node)
                break
    # print("There are {} lonely nodes".format(len(lonely_nodes)))
    graph.remove_nodes_from(lonely_nodes)
    return graph, len(lonely_nodes)



if __name__ == '__main__':
    file_name = "graph"
    times_bron_kerbosch = []
    clique_size_bron_kerbosch = []
    times_greedy = []
    clique_size_greedy = []
    times_color_greedy_ver1 = []
    clique_size_color_greedy_ver1 = []
    times_color_greedy_ver2 = []
    clique_size_color_greedy_ver2 = []
    times_color_greedy_ver3 = []
    clique_size_color_greedy_ver3 = []
    times_color_greedy_ver4 = []
    clique_size_color_greedy_ver4 = []
    times_color_greedy_ver5 = []
    clique_size_color_greedy_ver5 = []
    lonely_nodes = []
    for i in range(10, NAX_COLORS):
        graph_, node_to_label, label_to_node = nx.read_gpickle(file_name + "_{}_classes".format(i - 1))
        # remove lonely nodes
        graph_, n_lonely_nodes = remove_lonely_nodes(graph_, label_to_node, node_to_label)

        t1 = time.time()
        clique_bron_kerbosch = bron_kerbosch(graph_.copy(), node_to_label, label_to_node, [], list(graph_.nodes()),
                                             [])
        t2 = time.time()
        times_bron_kerbosch.append(t2 - t1)
        clique_size_bron_kerbosch.append(max([len(c) for c in clique_bron_kerbosch]))

        node2deg = {n: graph_.degree(n) for n in graph_.nodes}
        node_max_deg = max(node2deg, key=node2deg.get)
        clique_greedy = greedy([node_max_deg], graph_.copy(), graph_.nodes, label_to_node, node_to_label,
                               trip=False)
        triplets = [pair for pair in combinations(clique_greedy, 3)]
        for triplet in triplets:
            if len(clique_greedy) == i:
                break
            else:
                next_clique_greedy = greedy(list(triplet), graph_.copy(), graph_.nodes, label_to_node, node_to_label,
                                            trip=True)
            if len(next_clique_greedy) > len(clique_greedy):
                clique_greedy = next_clique_greedy
        clique_size_greedy.append((len(clique_greedy)))
        t3 = time.time()
        times_greedy.append(t3 - t2)


        # rank nodes and labels
        node_score, label_score = rank_nodes_ver1(graph_.copy(), label_to_node, node_to_label)
        sorted_labels = {k: v for k, v in sorted(label_score.items(), key=lambda item: item[1])}
        # sort nodes in label by their score
        label_to_node_ver1 = label_to_node.copy()
        for label in label_to_node:
            label_to_node_ver1[label] = sorted(label_to_node_ver1[label], key=lambda node: node_score[node])
        clique_new_algo_ver1 = color_greedy_ver1(graph_.copy(), node_to_label, label_to_node_ver1,
                                                  list(label_to_node.keys()), [],
                                                  list(graph_.nodes), 0)
        t4 = time.time()
        times_color_greedy_ver1.append(t4 - t3)
        clique_size_color_greedy_ver1.append(len(clique_new_algo_ver1))

        # new algorithm ver 2

        #remove half graph:
        new_graph = graph_.copy()
        [new_graph.remove_nodes_from(list(np.array(nodes_in_label)[np.argpartition([graph_.degree(n)
                                                                                 for n in nodes_in_label],
                                                                                int(np.ceil(len(nodes_in_label) / 2)))[:int(-np.ceil(len(nodes_in_label)/2))]])) for nodes_in_label in label_to_node.values()]
        nodes_dict = rank_nodes_ver2(new_graph, label_to_node, node_to_label)
        clique_new_algo_ver2 = color_greedy_ver2(new_graph, node_to_label, label_to_node, nodes_dict,
                                                  list(label_to_node.keys()), [], list(new_graph.nodes), [])
        t5 = time.time()
        times_color_greedy_ver2.append((t5 - t4))
        clique_size_color_greedy_ver2.append(len(clique_new_algo_ver2))

        clique_ver3 = color_greedy_ver3(graph_, node_to_label, label_to_node, list(label_to_node.keys()), [], list(graph_.nodes),
                           [])
        t6 = time.time()
        times_color_greedy_ver3.append(t6 - t5)
        clique_size_color_greedy_ver3.append(len(clique_ver3))

        clique_ver4 = color_greedy_ver4(graph_, node_to_label, label_to_node, list(label_to_node.keys()), [], list(graph_.nodes),
                           [])
        t7 = time.time()
        times_color_greedy_ver4.append(t7 - t6)
        clique_size_color_greedy_ver4.append(len(clique_ver4))

        clique_ver5 = color_greedy_ver5(graph_, node_to_label, label_to_node, list(label_to_node.keys()), [], list(graph_.nodes),
                           [])
        t8 = time.time()
        times_color_greedy_ver5.append(t8-t7)
        clique_size_color_greedy_ver5.append(len(clique_ver5))

#         plot results
visualization.plot_clique_size([clique_size_bron_kerbosch, clique_size_greedy, clique_size_color_greedy_ver1, clique_size_color_greedy_ver2
                                 ,clique_size_color_greedy_ver3, clique_size_color_greedy_ver4, clique_size_color_greedy_ver5],
                                ["Bron-Kerbosch", "Greedy", "Color Greedy ver1", "Color Greedy ver2", "Color Greedy ver3", "Color Greedy ver4", "Color Greedy ver5"])
visualization.plot_all_times([times_bron_kerbosch, times_greedy, times_color_greedy_ver1, times_color_greedy_ver2,
                                times_color_greedy_ver3, times_color_greedy_ver4, times_color_greedy_ver5],
                                 ["Bron-Kerbosch", "Greedy", "Color Greedy ver1", "Color Greedy ver2", "Color Greedy ver3", "Color Greedy ver4", "Color Greedy ver5"])

