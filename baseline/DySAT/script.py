import numpy as np
import os
# import dill
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn import datasets

from sklearn.model_selection import train_test_split
from utils.utilities import run_random_walks_n2v

import torch

np.random.seed(123)

def load_graphs(dataset_str):
    """Load graph snapshots given the name of dataset"""
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs] # Sparse matrix
    return graphs, adjs

def load_graphs_new(dataset_str):
    with open("data/{}/{}".format(dataset_str, "adj_time_list.pickle"), "rb") as handle:
        adj_time_list = pkl.load(handle,encoding="latin1")
    
    graphs = []
    for i in range(len(adj_time_list)):
        G = nx.from_scipy_sparse_matrix(adj_time_list[i], create_using=nx.MultiGraph)
        graphs.append(G)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs] # Sparse matrix
    # check = [np.sum(adj_time_list[j] != adjs[j]) for j in range(len(graphs))]
    # print(check)
    return graphs, adjs



if __name__ == "__main__":
    dataset = "Enron"
    time_steps = 16
    graphs, adjs = load_graphs(dataset)
    # print(len(adjs))
    # print(adjs[0])
    # print(graphs[0][0])
    # print([graph.number_of_nodes() for graph in graphs])

    graphs, adjs = load_graphs_new("enron10")
    # print([graph.number_of_nodes() for graph in graphs])
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx+1]
    print(eval_idx)