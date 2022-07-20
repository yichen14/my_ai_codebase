from argparse import ArgumentError
import os 
import pickle
import numpy as np

import torch
from torch_geometric.utils import dense_to_sparse, to_undirected, to_dense_adj
from torch_geometric.data import Data

from .load_utils import edge_tv_split, edge_tvt_split, random_attack_temporal

class TData(Data):
    TR = 0
    VA = 1
    TE = 2
    ALL = 3

    def __init__(self, **kwargs):
        super(TData, self).__init__(**kwargs)

        # Getter methods so I don't have to write this every time
        self.tr = lambda t : self.eis[t][:, self.masks[t][0]]
        self.va = lambda t : self.eis[t][:, self.masks[t][1]]
        self.te = lambda t : self.eis[t][:, self.masks[t][2]]
        self.all = lambda t : self.eis[t]

        # To match Euler models
        self.xs = self.x
        self.x_dim = self.x.size(1)

    def get_masked_edges(self, t, mask):
        if mask == self.TR:
            return self.tr(t)
        elif mask == self.VA:
            return self.va(t)
        elif mask == self.TE:
            return self.te(t)
        elif mask == self.ALL:
            return self.all(t)
        else:
            raise ArgumentError("Mask must be TData.TR, TData.VA, TData.TE, or TData.ALL")

    def ei_masked(self, mask, t):
        '''
        So method sig matches Euler models
        '''
        return self.get_masked_edges(t, mask)

    def ew_masked(self, *args):
        '''
        VGRNN datasets don't have weighted edges
        '''
        return None

'''
For loading datasets from the VRGNN repo (none have features)
'''
def load_vgrnn(dataset, ptb_rate, test_len):
    datasets = ['fb', 'dblp', 'enron10']
    assert dataset in datasets, \
        "Dataset %s not in allowed list: %s" % (dataset, str(datasets))

    adj = os.path.join('./data', dataset, "adj_time_list.pickle")
    with open(adj, 'rb') as f:
        fbytes = f.read() 

    adj_time_list = pickle.loads(fbytes, encoding='latin1')
    
    # Attack
    adj_time_list = random_attack_temporal(adj_time_list, ptb_rate, test_len)
    dense_adj_list = csr_matrix_to_tensor(adj_time_list)
    num_nodes = max([dense_adj_list[i].size(0) for i in range(len(dense_adj_list))])
    
    eis = []
    splits = []

    for adj in dense_adj_list:
        # Remove self loops
        for i in range(adj.size(0)):
            adj[i,i] = 0

        ei = dense_to_sparse(adj)[0]
        ei = to_undirected(ei)
        eis.append(ei)
        
        splits.append(edge_tvt_split(ei))


    data = TData(
        x=torch.eye(num_nodes),
        eis=eis,
        masks=splits,
        num_nodes=num_nodes,
        dynamic_feats=False,
        T=len(eis)
    )    

    return data 

def csr_matrix_to_tensor(matrices):
    adj_orig_dense_list = []
    for i in range(len(matrices)):
        data = matrices[i].tocoo()
        values = data.data
        indices = np.vstack((data.row, data.col))
        adj_orig_dense_list.append(torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(data.shape)).to_dense())
    return adj_orig_dense_list