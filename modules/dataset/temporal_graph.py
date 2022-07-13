import pickle
import numpy as np
import scipy.sparse as sp
import torch_geometric
import torch
from torch_geometric.data import Data
from utils import get_dataset_root
import attack
import os

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
        
        adj_time_list_path = os.path.join("./data", data_name, "adj_time_list.pickle")
        with open(adj_time_list_path, 'rb') as handle:
            self.adj_time_list = pickle.load(handle,encoding="latin1")

        if attack_flag and attack_func is not None:
            self.adj_time_list = attack_func(self.cfg, self.adj_time_list, self.device)

        self.time_step = len(self.adj_time_list)
        self.adj_orig_dense_list = csr_matrix_to_tensor(self.adj_time_list)
        self.num_nodes = self.adj_orig_dense_list[0].shape[0]
        
        if use_feat:
            feat_path = os.path.join(get_dataset_root(), data_name, "feat.npy")
            self.feat = np.load(feat_path)
        else:
            self.feat = [torch.tensor(np.eye(self.num_nodes).astype(np.float32)) for i in range(self.time_step)]
        self.feat_dim = self.feat[0].shape[1]

        self.data = [Data(x=self.feat[i], edge_index = self.adj_time_list[i]) for i in range(self.time_step)]
        self.pos_edges_l, self.neg_edges_l = self.mask_edges_prd()
        self.prepare_edge_list()

    def prepare_edge_list(self):
        edge_list = self.mask_edges_det(0.01, 0.01)[1]
        self.edge_idx_list = []
        for i in range(len(edge_list)):
            self.edge_idx_list.append(torch.tensor(np.transpose(edge_list[i]), dtype=torch.long))
    
    def mask_edges_det(self, val_ratio = 0.05, test_ratio = 0.10):
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
            edges_all = sparse_to_tuple(adj)[0]
            num_test = int(np.floor(edges.shape[0] * test_ratio))
            num_val = int(np.floor(edges.shape[0] * val_ratio))

            all_edge_idx = np.array(range(edges.shape[0]))
            np.random.shuffle(all_edge_idx)
            val_edge_idx = all_edge_idx[:num_val]
            test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
            test_edges = edges[test_edge_idx]
            val_edges = edges[val_edge_idx]
            train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

            edges_list.append(edges)

            def ismember(a, b, tol=5):
                rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
                return np.any(rows_close)

            test_edges_false = []
            while len(test_edges_false) < len(test_edges):
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])
                if idx_i == idx_j:
                    continue
                if ismember([idx_i, idx_j], edges_all):
                    continue
                if test_edges_false:
                    if ismember([idx_j, idx_i], np.array(test_edges_false)):
                        continue
                    if ismember([idx_i, idx_j], np.array(test_edges_false)):
                        continue
                test_edges_false.append([idx_i, idx_j])

            val_edges_false = []
            while len(val_edges_false) < len(val_edges):
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])
                if idx_i == idx_j:
                    continue
                if ismember([idx_i, idx_j], train_edges):
                    continue
                if ismember([idx_j, idx_i], train_edges):
                    continue
                if ismember([idx_i, idx_j], val_edges):
                    continue
                if ismember([idx_j, idx_i], val_edges):
                    continue
                if val_edges_false:
                    if ismember([idx_j, idx_i], np.array(val_edges_false)):
                        continue
                    if ismember([idx_i, idx_j], np.array(val_edges_false)):
                        continue
                val_edges_false.append([idx_i, idx_j])

            assert ~ismember(test_edges_false, edges_all)
            assert ~ismember(val_edges_false, edges_all)
            assert ~ismember(val_edges, train_edges)
            assert ~ismember(test_edges, train_edges)
            assert ~ismember(val_edges, test_edges)

            data = np.ones(train_edges.shape[0])

            # Re-build adj matrix
            adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
            adj_train = adj_train + adj_train.T

            adj_train_l.append(adj_train)
            train_edges_l.append(train_edges)
            val_edges_l.append(val_edges)
            val_edges_false_l.append(val_edges_false)
            test_edges_l.append(test_edges)
            test_edges_false_l.append(test_edges_false)

        # NOTE: these edge lists only contain single direction of edge!
        return adj_train_l, train_edges_l, val_edges_l, val_edges_false_l, test_edges_l, test_edges_false_l

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
            adj_tuple = sparse_to_tuple(adj_triu)
            edges = adj_tuple[0]
            edges_all = sparse_to_tuple(adj)[0]
            num_false = int(edges.shape[0])

            pos_edges_l.append(edges)

            def ismember(a, b, tol=5):
                rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
                return np.any(rows_close)

            edges_false = []
            while len(edges_false) < num_false:
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])
                if idx_i == idx_j:
                    continue
                if ismember([idx_i, idx_j], edges_all):
                    continue
                if edges_false:
                    if ismember([idx_j, idx_i], np.array(edges_false)):
                        continue
                    if ismember([idx_i, idx_j], np.array(edges_false)):
                        continue
                edges_false.append([idx_i, idx_j])

            assert ~ismember(edges_false, edges_all)

            false_edges_l.append(edges_false)

        # NOTE: these edge lists only contain single direction of edge!
        return pos_edges_l, false_edges_l

    def mask_edges_prd_new(self, val_ratio = 0.05, test_ratio = 0.10):
        pos_edges_l , false_edges_l = [], []
        edges_list = []
        adjs_list, adj_orig_dense_list = self.adjs_list, self.adj_orig_dense_list
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

        adj = adjs_list[0]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_false = int(edges.shape[0])

        pos_edges_l.append(edges)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])

        assert ~ismember(edges_false, edges_all)
        false_edges_l.append(np.asarray(edges_false))

        for i in range(1, len(adjs_list)):
            edges_pos = np.transpose(np.asarray(np.where((adj_orig_dense_list[i] - adj_orig_dense_list[i-1])>0)))
            num_false = int(edges_pos.shape[0])

            adj = adjs_list[i]
            # Remove diagonal elements
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            # Check that diag is zero:
            assert np.diag(adj.todense()).sum() == 0

            adj_triu = sp.triu(adj)
            adj_tuple = sparse_to_tuple(adj_triu)
            edges = adj_tuple[0]
            edges_all = sparse_to_tuple(adj)[0]

            edges_false = []
            while len(edges_false) < num_false:
                idx_i = np.random.randint(0, adj.shape[0])
                idx_j = np.random.randint(0, adj.shape[0])
                if idx_i == idx_j:
                    continue
                if ismember([idx_i, idx_j], edges_all):
                    continue
                if edges_false:
                    if ismember([idx_j, idx_i], np.array(edges_false)):
                        continue
                    if ismember([idx_i, idx_j], np.array(edges_false)):
                        continue
                edges_false.append([idx_i, idx_j])

            assert ~ismember(edges_false, edges_all)

            false_edges_l.append(np.asarray(edges_false))
            pos_edges_l.append(edges_pos)

        # NOTE: these edge lists only contain single direction of edge!
        return pos_edges_l, false_edges_l

    def sparse_to_tuple(self, sparse_mx):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape