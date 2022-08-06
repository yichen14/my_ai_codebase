import json
import numpy as np
import pandas as pd
import pickle
import torch
from deeprobust.graph.global_attack import Random

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

def csr_matrix_to_tensor(matrices):
    adj_orig_dense_list = []
    for i in range(len(matrices)):
        data = matrices[i].tocoo()
        values = data.data
        indices = np.vstack((data.row, data.col))
        adj_orig_dense_list.append(torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(data.shape)).to_dense())
    return adj_orig_dense_list

def preprocess(data_name, ptb_rate):

    # Load the adj matrix
    with open("data/{}/{}".format(data_name, "adj_time_list.pickle"), "rb") as handle:
        adj_time_list = pickle.load(handle,encoding="latin1")
    
    # Attack
    adj_time_list = random_attack_temporal(adj_time_list, ptb_rate, test_len=3)
    adj_orig_dense_list = csr_matrix_to_tensor(adj_time_list)

    # Process based on GTAT
    u_list, i_list, ts_list = [], [], []
    idx_list = []
    idx = 0

    for ts in range(len(adj_orig_dense_list)):
        adj_matrix = adj_orig_dense_list[ts]
        temp = torch.nonzero(adj_matrix)
        for j in range(len(temp)):
            u = int(temp[j][0])
            i = int(temp[j][1])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            idx_list.append(idx)

            idx = idx + 1

    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'idx':idx_list})


def run(data_name):
    OUT_DF = './processed/ml_{}.csv'.format(data_name)
    OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
    
    df = preprocess(data_name, 0.0)
    feat_dim = 16
    
    max_idx = max(df.u.max(), df.i.max())
    rand_feat = np.zeros((max_idx + 1, feat_dim))

    print(rand_feat.shape)
    df.to_csv(OUT_DF)
    np.save(OUT_NODE_FEAT, rand_feat)
    
run('enron10')