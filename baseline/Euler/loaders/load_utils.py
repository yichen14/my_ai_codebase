from time import time
import torch 
import numpy as np
from deeprobust.graph.global_attack import Random

'''
Splits edges into 85:5:10 train val test partition
(Following route of VGRNN paper)
'''
def edge_tvt_split(ei):
    ne = ei.size(1)
    val = int(ne*0.85)
    te = int(ne*0.90)

    masks = torch.zeros(3, ne).bool()
    rnd = torch.randperm(ne)
    masks[0, rnd[:val]] = True 
    masks[1, rnd[val:te]] = True
    masks[2, rnd[te:]] = True 

    return masks[0], masks[1], masks[2]

'''
For the cyber data, all of the test set is the latter time
stamps. So only need train and val partitions
'''
def edge_tv_split(ei, v_size=0.05):
    ne = ei.size(1)
    val = int(ne*v_size)

    masks = torch.zeros(2, ne).bool()
    rnd = torch.randperm(ne)
    masks[1, rnd[:val]] = True
    masks[0, rnd[val:]] = True 

    return masks[0], masks[1]

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
    