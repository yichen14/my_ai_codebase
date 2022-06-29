from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch

class GAE(torch.nn.Module):
    def __init__(self, nfeat, nhid, out_dim):
        super(GAE, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, out_dim)

    def encode(self, x, train_pos_edge_index):
        out = self.conv1(x, train_pos_edge_index) # convolution 1
        out = out.relu()
        return self.conv2(out, train_pos_edge_index) # convolution 2

    def decode(self, z, edge_label_index): # only pos and neg edges
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z): 
        prob_adj = z @ z.t() # get adj NxN
        return (prob_adj > 0).nonzero(as_tuple=False).t()