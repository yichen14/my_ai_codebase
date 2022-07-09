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

"""

class RecurrentGCN_EGCNH(torch.nn.Module):
    """
    Arg:
        num_nodes (int): Number of vertices. 
        in_channels (int): Number of filters.
    """
    def __init__(self, cfg):
        super(RecurrentGCN_EGCNH, self).__init__()
        in_channels = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        num_nodes = cfg.TASK_SPECIFIC.GEOMETRIC.num_nodes
        inner_prod = cfg.TASK_SPECIFIC.GEOMETRIC.inner_prod

        self.recurrent = rcrgcn.EvolveGCNH(num_nodes, in_channels)
        
        if not inner_prod:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_channels*2, 1) # Undirected? 
                torch.nn.Sigmoid()
            )
        else:
            self.classifier = None

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.classifier(h)
        return h

class RecurrentGCN_EGCNO(torch.nn.Module):
    def __init__(self, cfg):
        super(RecurrentGCN_EGCNO, self).__init__()
        in_channels = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        inner_prod = cfg.TASK_SPECIFIC.GEOMETRIC.inner_prod

        self.recurrent = rcrgcn.EvolveGCNO(in_channels)

        if not inner_prod:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_channels*2, in_channels*2) # Undirected? 
                torch.nn.Sigmoid()
            )
        else:
            self.classifier = None
    
    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.classifier(h)
        return h


