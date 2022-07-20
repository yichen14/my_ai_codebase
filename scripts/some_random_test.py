import numpy as np

# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# dataset_module = Planetoid(root = "/home/randomgraph/data/", name="Cora")
# transform = T.Compose([
#     T.ToUndirected(merge = True),
#     T.ToDevice("cpu"),
#     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                 split_labels=True, add_negative_train_samples=False),
# ]) 

# data = transform(dataset_module[0])
# print(dataset_module.adj)
# train_data, val_data, test_data = data
# print(train_data.edge_index)
# print(train_data.pos_edge_label_index)


# from torch_geometric_temporal.signal import temporal_signal_split
# from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
# loader = ChickenpoxDatasetLoader()
# data = loader.get_dataset()
# train_data, test_data = temporal_signal_split(data, train_ratio=0.2)
# for time, snapshot in enumerate(train_data):
#     print(snapshot.edge_attr, snapshot.edge_index)
#     break

from deeprobust.graph.data import Dataset
from deeprobust.graph.global_attack import Random
ptb_rate = 0.5
data = Dataset(root='/tmp/', name='cora')
adj, features, labels = data.adj, data.features, data.labels
num_edges = np.sum(adj)
num_perturbations = int(num_edges*ptb_rate)//2
print(num_edges) # 10138
print(num_perturbations) # 2534
model = Random()
model.attack(adj, n_perturbations=num_perturbations)
modified_adj = model.modified_adj
print(np.sum(modified_adj)) #15206