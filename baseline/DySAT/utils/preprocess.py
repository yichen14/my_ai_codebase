
import numpy as np
# import dill
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from utils.utilities import run_random_walks_n2v

from deeprobust.graph.global_attack import Random

np.random.seed(123)

def load_graphs(dataset_str):
    """Load graph snapshots given the name of dataset"""
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]
    return graphs, adjs

def get_context_pairs(graphs, adjs):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    print("Computing training pairs ...")
    context_pairs_train = []
    for i in range(len(graphs)):
        context_pairs_train.append(run_random_walks_n2v(graphs[i], adjs[i], num_walks=10, walk_len=20))

    return context_pairs_train

def get_evaluation_data(graphs):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = len(graphs) - 2
    eval_graph = graphs[eval_idx] # Last second for val
    next_graph = graphs[eval_idx+1] # Last one for test
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2, 
                            test_mask_fraction=0.6)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.6):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):
            edges_positive.append(e)
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)
    

    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive, 
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos, 
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg
            
def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg

"""
Load enron10, fb, dblp
"""
def load_graphs_new(dataset_str, ptb_rate, test_len):
    with open("data/{}/{}".format(dataset_str, "adj_time_list.pickle"), "rb") as handle:
        adj_time_list = pkl.load(handle,encoding="latin1")
    
    # Attack
    adj_time_list = random_attack_temporal(adj_time_list, ptb_rate, test_len)

    # Conver sparse matrix to MultiGraph
    graphs = []
    for i in range(len(adj_time_list)):
        G = nx.from_scipy_sparse_matrix(adj_time_list[i], create_using=nx.MultiGraph)
        graphs.append(G)

    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs] # Sparse matrix
    return graphs, adjs

"""
Random Attack
"""
def random_attack_temporal(adj_matrix_lst, ptb_rate, test_len):
    random_attack = Random()
    
    attack_data = []
    for time_step in range(len(adj_matrix_lst)-test_len):
        num_edges = np.sum(adj_matrix_lst[time_step])
        num_modified = int(num_edges*ptb_rate)//2
        adj_matrix = adj_matrix_lst[time_step]
        random_attack.attack(adj_matrix, n_perturbations = num_modified, type="add")
        attack_data.append(random_attack.modified_adj)
    
    for time_step in range(len(adj_matrix_lst) - test_len, len(adj_matrix_lst)):
        attack_data.append(adj_matrix_lst[time_step])

    return attack_data