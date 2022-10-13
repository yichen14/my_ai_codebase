from concurrent.futures import process
import numpy as np
import pandas as pd
import os
import logging
import pickle
import random

import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch_geometric
from sklearn import preprocessing
from deeprobust.graph.global_attack import Random

# from interioir 
import attack
from utils.misc import get_dataset_root, get_process_root
from utils.tgat_graph import NeighborFinder
from utils.tgat_utils import RandEdgeSampler

"""
Helper functions
"""
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def csr_matrix_to_tensor(matrices):
    adj_orig_dense_list = []
    for i in range(len(matrices)):
        data = matrices[i].tocoo()
        values = data.data
        indices = np.vstack((data.row, data.col))
        adj_orig_dense_list.append(torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(data.shape)).to_dense())
    return adj_orig_dense_list

def to_undirect(sparse_matrices):
    dense_matrices = csr_matrix_to_tensor(sparse_matrices)
    undirect_dense_list = []
    undirect_sparse_list = []
    # N = dense_matrices[0].shape[0]
    for matrix in dense_matrices:
        # for i in range(N):
        #     for j in range(N):
        #         if matrix[i, j] == 1:
        #             matrix[j, i] = 1
        matrix = torch.logical_or(matrix, matrix.T)
        undirect_dense_list.append(matrix)
        undirect_sparse_list.append(csr_matrix(np.array(matrix.tolist())))
    return undirect_dense_list, undirect_sparse_list

## Continuous temporal graph process
"""
Modified from TGAT: 
https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
"""
class continuous_temporal_graph(torch_geometric.data.Dataset):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device

        data_name = self.cfg.DATASET.dataset
        use_edge_feat = cfg.DATASET.TEMPORAL.use_feat
        attack_flag = True if cfg.ATTACK.method != "none" else False
        attack_func = attack.dispatcher(cfg)

        self.val_len = self.cfg.DATASET.TEMPORAL.val_len
        self.test_len = self.cfg.DATASET.TEMPORAL.test_len - self.val_len

        self.lr = self.cfg.TRAIN.initial_lr
        self.ptb_rate = self.cfg.ATTACK.ptb_rate
        self.attack_method = self.cfg.ATTACK.method

        # generate g_df, n_feat, e_feat
        self.run(data_name, use_edge_feat, attack_flag, attack_func)

        process_path = os.path.join(get_process_root())
        self.g_df = pd.read_csv('{}/ml_{}_{}_{}_lr_{}.csv'.format(process_path, data_name, self.attack_method, self.ptb_rate, self.lr))
        self.n_feat = np.load('{}/ml_{}_{}_{}_lr_{}_node.npy'.format(process_path, data_name, self.attack_method, self.ptb_rate, self.lr))
        self.e_feat = np.load('{}/ml_{}_{}_{}_lr_{}.npy'.format(process_path, data_name, self.attack_method, self.ptb_rate, self.lr))

        self.load_data()

    def preprocess(self, data_name, attack_flag = False, attack_func = None):

        adj_time_list_path = os.path.join(get_dataset_root(), data_name, "adj_time_list.pickle")
        with open(adj_time_list_path, 'rb') as handle:
            self.adj_time_list = pickle.load(handle,encoding="latin1")

        # self.adj_orig_dense_list, self.adj_time_list = to_undirect(self.adj_time_list) # to undirect

        # Attack 
        if attack_flag and attack_func is not None:
            self.adj_time_list = attack_func(self.cfg, self.adj_time_list, self.device)
       
        self.adj_orig_dense_list = csr_matrix_to_tensor(self.adj_time_list)

        ## Process based on TGAT
        u_list, i_list, ts_list, idx_list = [], [], [], []
        
        idx = 0
        for ts in range(len(self.adj_orig_dense_list)):
            adj_matrix = self.adj_orig_dense_list[ts]
            temp = torch.nonzero(adj_matrix)
            for j in range(len(temp)):
                u = int(temp[j][0])
                i = int(temp[j][1])

                # if u in u_list and i in i_list:
                #     continue

                u_list.append(u)
                i_list.append(i)
                ts_list.append(ts)
                idx_list.append(idx)
                idx = idx + 1

        return pd.DataFrame({'u': u_list, 
                            'i':i_list, 
                            'ts':ts_list, 
                            'idx':idx_list})

    def reindex(self, df):
        upper_u = df.u.max() + 1
        # new_i = df.i + upper_u
        
        new_df = df.copy()
        # print(new_df.u.max())
        # print(new_df.i.max())
        
        # new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
        new_df.ts += 1
        
        print("upper src", new_df.u.max())
        print("upper dst", new_df.i.max())
        print("upper idx", new_df.idx.max())
        
        return new_df

    def run(self, data_name, use_edge_feat, attack_flag = False, attack_func = None):
        process_path = os.path.join(get_process_root())
        OUT_DF = '{}/ml_{}_{}_{}_lr_{}.csv'.format(process_path, data_name, self.attack_method, self.ptb_rate, self.lr)
        OUT_NODE_FEAT = '{}/ml_{}_{}_{}_lr_{}_node.npy'.format(process_path, data_name, self.attack_method, self.ptb_rate, self.lr)
        OUT_FEAT = '{}/ml_{}_{}_{}_lr_{}.npy'.format(process_path, data_name, self.attack_method, self.ptb_rate, self.lr)
        print(OUT_DF)
        
        df = self.preprocess(data_name, attack_flag, attack_func)
        new_df = self.reindex(df)

        """
        From MetaDyGNN process_dblp.dy
        """
        feat_dim = 172
        if use_edge_feat:
            feat = torch.empty((new_df.idx.max() + 1, feat_dim))
            feat = torch.nn.init.xavier_uniform_(feat, gain=1.414)
            print("Edge feature", feat.shape)
        else: 
            feat = None

        max_idx = max(new_df.u.max(), new_df.i.max())
        # rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
        rand_feat = torch.empty((max_idx + 1, feat.shape[1]))
        rand_feat = torch.nn.init.xavier_uniform_(rand_feat)
        print("Node feature", rand_feat.shape)

        new_df.to_csv(OUT_DF)
        np.save(OUT_NODE_FEAT, rand_feat)
        np.save(OUT_FEAT, feat)
    
    def load_data(self):
        # Time split
        print(sorted(set(self.g_df.ts)))
        self.val_time = sorted(set(self.g_df.ts))[-self.val_len-self.test_len]
        self.test_time = sorted(set(self.g_df.ts))[-self.test_len]
        print("val_time {}, test_time {}".format(self.val_time, self.test_time))

        self.src_l = self.g_df.u.values
        self.dst_l = self.g_df.i.values
        self.e_idx_l = self.g_df.idx.values
        self.ts_l = self.g_df.ts.values

        max_idx = max(self.src_l.max(), self.dst_l.max())
        max_src_index = self.src_l.max()

        print('max node:', max_idx)

        # Data split
        random.seed(2022)

        self.node_set = set(np.unique(np.hstack([self.g_df.u.values, self.g_df.i.values])))
        self.total_unique_nodes = len(self.node_set)

        # training set
        self.valid_train_flag = (self.ts_l < self.val_time)
        
        self.train_src_l = self.src_l[self.valid_train_flag]
        self.train_dst_l = self.dst_l[self.valid_train_flag]
        self.train_ts_l = self.ts_l[self.valid_train_flag]
        self.train_e_idx_l = self.e_idx_l[self.valid_train_flag]

        # validation and testing sets
        self.valid_val_flag = (self.ts_l < self.test_time) * (self.ts_l >= self.val_time)
        self.valid_test_flag = (self.ts_l >= self.test_time)

        self.val_src_l = self.src_l[self.valid_val_flag]
        self.val_dst_l = self.dst_l[self.valid_val_flag]
        self.val_ts_l = self.ts_l[self.valid_val_flag]
        self.val_e_idx_l = self.e_idx_l[self.valid_val_flag]

        self.test_src_l = self.src_l[self.valid_test_flag]
        self.test_dst_l = self.dst_l[self.valid_test_flag]
        self.test_ts_l = self.ts_l[self.valid_test_flag]
        self.test_e_idx_l = self.e_idx_l[self.valid_test_flag]

        ### Initialize the data structure for graph and edge sampling
        # build the graph for fast query
        # graph only contains the training data (with 10% nodes removal)
        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(self.train_src_l, self.train_dst_l, self.train_e_idx_l, self.train_ts_l):
            adj_list[src].append((dst, eidx, ts))
            adj_list[dst].append((src, eidx, ts))
        self.train_ngh_finder = NeighborFinder(adj_list)

        # full graph with all the data for the test and validation purpose
        full_adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(self.src_l, self.dst_l, self.e_idx_l, self.ts_l):
            full_adj_list[src].append((dst, eidx, ts))
            full_adj_list[dst].append((src, eidx, ts))
        self.full_ngh_finder = NeighborFinder(full_adj_list)

        self.train_rand_sampler = RandEdgeSampler(self.train_src_l, self.train_dst_l)
        self.val_rand_sampler = RandEdgeSampler(self.src_l, self.dst_l)
        self.test_rand_sampler = RandEdgeSampler(self.src_l, self.dst_l)

        