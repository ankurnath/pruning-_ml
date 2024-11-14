import pickle
import networkx as nx
import os
from collections import defaultdict
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool

from torch_geometric.data import Data, Batch
def load_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    print(f'Data has been loaded from {file_path}')
    return loaded_data


def save_to_pickle(data, file_path):
    """
    Save data to a pickle file.

    Parameters:
    - data: The data to be saved.
    - file_path: The path to the pickle file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Data has been saved to {file_path}')


def make_subgraph(graph, nodes):
    assert type(graph) == nx.Graph
    subgraph = nx.Graph()
    edges_to_add = []
    for node in nodes:
        edges_to_add += [(u, v) for u, v in list(graph.edges(node))]
    subgraph.add_edges_from(edges_to_add)
    return subgraph


def relabel_graph(graph: nx.Graph, return_reverse_transformation_dic=False, return_forward_transformation_dic=False):
    """
    forward transformation has keys being original nodes and values being new nodes
    reverse transformation has keys being new nodes and values being old nodes
    """
    nodes = graph.nodes()
    n = graph.number_of_nodes()
    desired_labels = set([i for i in range(n)])
    already_labeled = set([int(node) for node in nodes if node < n])
    desired_labels = desired_labels - already_labeled
    transformation = {}
    reverse_transformation = {}
    for node in nodes:
        if node >= graph.number_of_nodes():
            transformation[node] = desired_labels.pop()
            reverse_transformation[transformation[node]] = node

    if return_reverse_transformation_dic and return_forward_transformation_dic:
        return nx.relabel_nodes(graph, transformation), transformation, reverse_transformation

    elif return_forward_transformation_dic:
        return nx.relabel_nodes(graph, transformation), transformation

    elif return_reverse_transformation_dic:
        return nx.relabel_nodes(graph, transformation), reverse_transformation

    else:
        return nx.relabel_nodes(graph, transformation)


