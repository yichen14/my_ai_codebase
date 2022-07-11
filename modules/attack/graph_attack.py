from deeprobust.graph.global_attack import Random
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
import os
import torch

def generate_random_attack(data, randomness, device):
    """
        perform a random attack on data
        Args:
            data: a graph data in Pytorch Geometric Dataset
            randomness: (float) percentage of edges need to be modified
            seed:(int) random seed
        Return:
            modified_data: (torch_geometric.datasets.Data)
    """
    data_dpr = Pyg2Dpr(data) # convert pyg data to dpr data

    adj, _ = data_dpr.adj, data_dpr.features

    num_edges = data_dpr.adj.shape[0]

    num_modified = int(num_edges*randomness)//3

    random_attack = Random(device=device)
    random_attack.attack(adj, n_perturbations = num_modified, type="add")
    adj = random_attack.modified_adj
    random_attack.attack(adj, n_perturbations = num_modified, type="remove")
    adj = random_attack.modified_adj
    random_attack.attack(adj, n_perturbations = num_modified, type="flip")

    data_dpr.adj = random_attack.modified_adj

    data_pyg = Dpr2Pyg(data_dpr)
    #  print(data_pyg.data)
    return data_pyg
