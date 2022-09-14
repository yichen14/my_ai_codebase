from operator import neg
import pickle
import numpy as np
import scipy.sparse as sp
import torch_geometric
import torch
from torch_geometric.data import Data
from utils import get_dataset_root
from utils.merge_graph import merge_graph
import attack
import os
import logging
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange
import logging
import networkx as nx
import datetime
from torch_geometric.utils import negative_sampling

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def csr_matrix_to_tensor(matrices, num_nodes):
    adj_orig_dense_list = []
    for i in range(len(matrices)):
        data = torch.zeros(num_nodes, num_nodes)
        graph = matrices[i].tocoo()
        for row_, col_ in zip(graph.row, graph.col):
            data[row_,col_] = 1
        adj_orig_dense_list.append(data)
    return adj_orig_dense_list

def to_undirect(dense_matrices):
    # dense_matrices = csr_matrix_to_tensor(sparse_matrices)
    undirect_dense_list = []
    undirect_sparse_list = []
    # N = dense_matrices[0].shape[0]
    for item in tqdm(dense_matrices):
        matrix = torch.Tensor(item)
        matrix = torch.logical_or(matrix, matrix.T).float()
        # matrix = torch.logical_or(matrix, torch.eye(len(matrix))).float()
        undirect_dense_list.append(matrix)
        undirect_sparse_list.append(csr_matrix(np.array(matrix.tolist())))
    return undirect_dense_list, undirect_sparse_list

# Temporal Graph
class temporal_graph(torch_geometric.data.Dataset):
    
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device

        data_name = self.cfg.DATASET.dataset
        use_feat = cfg.DATASET.TEMPORAL.use_feat
        attack_flag = True if cfg.ATTACK.method != "none" else False

        attack_func = attack.dispatcher(cfg)

        self.prepare(data_name, use_feat, attack_flag, attack_func)

    def prepare(self, data_name, use_feat, attack_flag = False, attack_func = None):
        
        adj_time_list_path = os.path.join(get_dataset_root(), data_name, "adj_time_list.pickle")
        with open(adj_time_list_path, 'rb') as handle:
            self.adj_time_list = pickle.load(handle,encoding="bytes")
            
        assert self.adj_time_list[0].max() == 1.0

        adj_orig_dense_list_path = os.path.join(get_dataset_root(), data_name, "adj_orig_dense_list.pickle")
        with open(adj_orig_dense_list_path, 'rb') as handle:
            self.adj_orig_dense_list = pickle.load(handle,encoding="bytes")

        self.num_nodes = self.gen_node_number(self.adj_time_list)
        # self.adj_orig_dense_list = csr_matrix_to_tensor(self.adj_time_list, self.num_nodes)
        # self.adj_orig_dense_list, self.adj_time_list = to_undirect(self.adj_orig_dense_list) # to undirect
        # self.adj_time_list = to_undirect(self.adj_time_list) # to undirect
        # self.adj_orig_dense_list, self.adj_time_list = to_undirect(self.adj_time_list) # to undirect

        # Attack 
        logging.info("Start to attack graphs, time:{}".format(datetime.datetime.now()))
        if attack_flag and attack_func is not None:
            self.adj_time_list = attack_func(self.cfg, self.adj_time_list, self.device)
        
        # For DySAT
        # Conver sparse matrix to MultiGraph
        if self.cfg.MODEL.model == "DYSAT":
            self.graphs = []
            for i in range(len(self.adj_time_list)):
                G = nx.from_scipy_sparse_matrix(self.adj_time_list[i], create_using=nx.MultiGraph)
                self.graphs.append(G)

        self.time_step = len(self.adj_time_list)
        self.adj_orig_dense_list = csr_matrix_to_tensor(self.adj_time_list, self.num_nodes)
        
        if use_feat and os.path.exists(os.path.join(get_dataset_root(), data_name, "feat.npy")):
            feat_path = os.path.join(get_dataset_root(), data_name, "feat.npy")
            self.feat = np.load(feat_path)
        else:
            self.feat = [torch.tensor(np.eye(self.num_nodes).astype(np.float32)) for i in range(self.time_step)]
        self.feat_dim = self.feat[0].shape[1]

        self.data = [Data(x=self.feat[i], edge_index = self.adj_time_list[i]) for i in range(self.time_step)]
        logging.info("Start to prepare edge list, time:{}".format(datetime.datetime.now()))
        self.pos_edges_l, self.neg_edges_l = self.mask_edges_prd()
        self.prepare_edge_list()
        logging.info("Finish to load temporal graphs, time:{}".format(datetime.datetime.now()))
        if self.cfg.MODEL.model in ['GAE', 'VGAE', "ProGCN", "RGCN"]:
            # if the model is GAE or any static graph neural network, merged dataset for static gnn training
            self.prepare_static_dataset()

    def gen_node_number(self, edge_lists):
        num_nodes = 0
        for i in range(len(edge_lists)):
            data = edge_lists[i].tocoo()
            index = max([data.row.max(), data.col.max()]) + 1
            if index > num_nodes:
                num_nodes = index  
        return num_nodes

    def load_from_data_dict(self, data_dict):
        self.adj_dense_merge_train = data_dict['adj_dense_merge_train']
        self.adj_dense_merge_test = data_dict['adj_dense_merge_test']
        self.adj_dense_merge_val = data_dict['adj_dense_merge_val']
        self.adj_sparse_merge_train = data_dict['adj_sparse_merge_train']
        self.adj_sparse_merge_test = data_dict['adj_sparse_merge_test']
        self.adj_sparse_merge_val = data_dict['adj_sparse_merge_val']
        self.feat_static_train = data_dict['feat_static_train']
        self.feat_static_test = data_dict['feat_static_test']
        self.feat_static_val = data_dict['feat_static_val']
        self.edge_idx_train = data_dict['edge_idx_train']
        self.edge_idx_test = data_dict['edge_idx_test']
        self.edge_idx_val = data_dict['edge_idx_val']
        self.pos_edges_l_static_train = data_dict['pos_edges_l_static_train']
        self.pos_edges_l_static_test = data_dict['pos_edges_l_static_test']
        self.pos_edges_l_static_val = data_dict['pos_edges_l_static_val']

    def prepare_static_dataset(self):

        # TODO: load data from local
        data_name = self.cfg.DATASET.dataset
        static_data_path = self.cfg.DATASET.STATIC.merged_data_path
        attack_data_path = self.cfg.ATTACK.attack_data_path
        ptb_rate = self.cfg.ATTACK.ptb_rate
        val_len = self.cfg.DATASET.TEMPORAL.val_len
        test_len = self.cfg.DATASET.TEMPORAL.test_len
        attack_method = self.cfg.ATTACK.method
        data_path = os.path.join(get_dataset_root(), attack_data_path, "{}_ptb_rate_{}_node".format(self.cfg.DATASET.dataset, ptb_rate))
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        pickle_path = os.path.join(data_path, "merged_data_ptb_{}_test_{}_{}_remove.pickle".format(ptb_rate,test_len, attack_method))

        data_dict={}

        if os.path.exists(pickle_path):
            logging.info("Load static data from merged_data_ptb_{}_test_{}_{}_remove.pickle".format(ptb_rate, test_len, attack_method))
            with open(pickle_path, 'rb') as handle:
                data_dict = pickle.load(handle,encoding="latin1")
        else:
            logging.info("Start merging dataset: {}".format(data_name))
            length = len(self.adj_time_list)

            data_dict['adj_dense_merge_train'] = merge_graph(self.adj_orig_dense_list[:length-test_len])
            data_dict['adj_sparse_merge_train'] = csr_matrix(data_dict['adj_dense_merge_train'].numpy())
            
            data_dict['adj_dense_merge_test']= merge_graph(self.adj_orig_dense_list[length-test_len+val_len:])
            data_dict['adj_sparse_merge_test'] = csr_matrix(np.array(data_dict['adj_dense_merge_test'].tolist()))

            data_dict['adj_dense_merge_val'] = merge_graph(self.adj_orig_dense_list[length-test_len:length-test_len+val_len])
            data_dict['adj_sparse_merge_val']= csr_matrix(np.array(data_dict['adj_dense_merge_val'].tolist()))

            def get_pos_neg_edge_lst(dense_matrix):
                pos_edge_lst = []
                for i in trange(dense_matrix.shape[0]):
                    for j in range(dense_matrix.shape[0]):
                        if dense_matrix[i, j] == 1.0 and [j, i] not in pos_edge_lst:
                            pos_edge_lst.append([i, j])
                return pos_edge_lst

            pos_edges_l_static_train = get_pos_neg_edge_lst(data_dict['adj_dense_merge_train'])
            pos_edges_l_static_test = get_pos_neg_edge_lst(data_dict['adj_dense_merge_test'])
            pos_edges_l_static_val = get_pos_neg_edge_lst(data_dict['adj_dense_merge_val'])

            data_dict['feat_static_train'] = merge_graph(self.feat[:length-test_len])
            data_dict['feat_static_test'] = merge_graph(self.feat[length-test_len+val_len:])
            data_dict['feat_static_val'] = merge_graph(self.feat[length-test_len:length-test_len+val_len])

            data_dict['edge_idx_train'] = torch.tensor(np.transpose(pos_edges_l_static_train), dtype=torch.long)
            data_dict['edge_idx_test'] = torch.tensor(np.transpose(pos_edges_l_static_test), dtype=torch.long)
            data_dict['edge_idx_val'] = torch.tensor(np.transpose(pos_edges_l_static_val), dtype=torch.long)
            
            data_dict['pos_edges_l_static_train'] = torch.tensor(pos_edges_l_static_train).T
            data_dict['pos_edges_l_static_test'] = torch.tensor(pos_edges_l_static_test).T
            data_dict['pos_edges_l_static_val'] = torch.tensor(pos_edges_l_static_val).T

            with open(pickle_path, 'ab') as handle:
                pickle.dump(data_dict, handle)

        self.load_from_data_dict(data_dict)

    def prepare_edge_list(self):
        edge_list = self.mask_edges_det()
        self.edge_idx_list = []
        for i in range(len(edge_list)):
            self.edge_idx_list.append(torch.tensor(np.transpose(edge_list[i]), dtype=torch.long))
            
    def mask_edges_det(self):
        adj_train_l, train_edges_l, val_edges_l = [], [], []
        val_edges_false_l, test_edges_l, test_edges_false_l = [], [], []
        edges_list = []
        adjs_list = self.adj_time_list

        for i in range(0, len(adjs_list)):
            # Function to build test set with 10% positive links
            # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

            adj = adjs_list[i]
            # Remove diagonal elements
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            # Check that diag is zero:
            assert np.diag(adj.todense()).sum() == 0

            adj_triu = sp.triu(adj)
            adj_tuple = sparse_to_tuple(adj_triu)
            edges = adj_tuple[0]
            # edges_all = sparse_to_tuple(adj)[0]

            train_edges = edges

            edges_list.append(edges)

            train_edges_l.append(train_edges)

        # NOTE: these edge lists only contain single direction of edge!
        return train_edges_l

    def mask_edges_prd(self):
        pos_edges_l , false_edges_l = [], []
        edges_list = []
        adjs_list = self.adj_time_list
        for i in range(0, len(adjs_list)):
            # Function to build test set with 10% positive links
            # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

            adj = adjs_list[i]

            # Remove diagonal elements
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            # Check that diag is zero:
            assert np.diag(adj.todense()).sum() == 0

            adj_triu = sp.triu(adj)
            adj_tuple = sparse_to_tuple(adj)
            edges = adj_tuple[0]
            pos_edges_l.append(edges)
            false_edges_l.append(negative_sampling(torch.Tensor(edges.T), adj.shape[0]).numpy().T)
            # false_edges_l.append(None)

        # NOTE: these edge lists only contain single direction of edge!
        return pos_edges_l, false_edges_l