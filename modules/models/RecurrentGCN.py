import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNH

class RecurrentGCN(torch.nn.Module):
    def __init__(self, cfg):
        super(RecurrentGCN, self).__init__()
        node_features = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
        node_count = cfg.TASK_SPECIFIC.GEOMETRIC.node_count
        self.recurrent = EvolveGCNH(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h