from signal import Sigmasks
import torch
import torch.nn.functional as F
import torch_geometric_temporal.nn.recurrent as rcrgcn

####################################################################################################################
# models from PyTorch Geometric Temporal
####################################################################################################################

"""
Recurrent GCN Model
1. EvolveGCNH: Evolving Graph Convolutional Hidden Layer.
Temporal Layer: GRU
GNN Layer: GCN

2. EvolveGCNO: Evolving Graph Convolutional without Hidden Layer.
Temporal Layer: LSTM
GNN Layer: GCN 

3. DCRNN: Diffusion Convolutional Gated Recurrent Unit. 
Temporal Layer: GRU
GNN Layer: DiffConv

4. GCLSTM: Integrated Graph Convolutional Long Short Term Memory Cell
Temporal Layer: LSTM
GNN Layer: Chebyshev

"""

class RecurrentGCN_EGCNH(torch.nn.Module):
    def __init__(self, cfg):
        super(RecurrentGCN_EGCNH, self).__init__()
        in_channels = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        num_nodes = cfg.TASK_SPECIFIC.GEOMETRIC.num_nodes
        # inner_prod = cfg.TASK_SPECIFIC.GEOMETRIC.inner_prod

        self.recurrent = rcrgcn.EvolveGCNH(num_nodes, in_channels)
        self.linear = torch.nn.Linear(in_channels, 1)
        

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.classifer(h)
        return h

class RecurrentGCN_EGCNO(torch.nn.Module):
    def __init__(self, cfg):
        super(RecurrentGCN_EGCNO, self).__init__()
        in_channels = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        inner_prod = cfg.TASK_SPECIFIC.GEOMETRIC.inner_prod

        self.recurrent = rcrgcn.EvolveGCNO(in_channels)
        
        if not inner_prod:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_channels, 1), 
                # torch.nn.Sigmoid()
            )
        else:
            self.classifier = None
    
    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.classifier(h)
        return h

class RecurrentGCN_DCRNN(torch.nn.Module):
    def __init__(self, cfg):
        super(RecurrentGCN_DCRNN, self).__init__()
        in_channels = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        out_channels = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        inner_prod = cfg.TASK_SPECIFIC.GEOMETRIC.inner_prod
        K = cfg.TASK_SPECIFIC.GEOMETRIC.filter_size 

        self.recurrent = rcrgcn.DCRNN(in_channels, out_channels, K)

        if not inner_prod:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_channels, 1), 
                # torch.nn.Sigmoid()
            )
        else:
            self.classifier = None
    
    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.classifier(h)
        return h

class RecurrentGCN_GCLSTM(torch.nn.Module):
    def __init__(self, cfg):
        super(RecurrentGCN_GCLSTM, self).__init__()
        in_channels = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        out_channels = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        inner_prod = cfg.TASK_SPECIFIC.GEOMETRIC.inner_prod
        K = cfg.TASK_SPECIFIC.GEOMETRIC.filter_size

        self.recurrent = rcrgcn.GCLSTM(in_channels, out_channels, K)
        if not inner_prod:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_channels, 1), 
                # torch.nn.Sigmoid()
            )
        else:
            self.classifier = None
    
    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.classifier(h)
        return h, h_0, c_0


