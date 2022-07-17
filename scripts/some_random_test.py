from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
dataset_module = Planetoid(root = "/home/randomgraph/data/", name="Cora")
print(dataset_module[0].edge_index)