import numpy as np
import torch
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
#     print(snapshot.y)
#     print(snapshot.y.size())
#     break


# from deeprobust.graph.data import Dataset
# from deeprobust.graph.global_attack import Random
# ptb_rate = 0.5
# data = Dataset(root='/tmp/', name='cora')
# adj, features, labels = data.adj, data.features, data.labels
# print(len(adj))
# num_edges = np.sum(adj)
# num_perturbations = int(num_edges*ptb_rate)//2
# print(num_edges) # 10138
# print(num_perturbations) # 2534
# model = Random()
# model.attack(adj, n_perturbations=num_perturbations)
# modified_adj = model.modified_adj
# print(np.sum(modified_adj)) #15206


# matrix = torch.tensor(np.array( [[0,0,0,0,1],
#                    [1,0,0,0,0],
#                    [0,1,0,1,0],
#                    [0,1,0,0,1],
#                    [0,0,0,0,0]]))

# print(torch.logical_or(matrix, matrix.T))

from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox
from deeprobust.graph.utils import preprocess
from deeprobust.graph.global_attack import NodeEmbeddingAttack
data = Dataset(root='/tmp/', name='cora')
adj, features, labels = data.adj, data.features, data.labels
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
print(idx_train.shape)
idx_unlabeled = np.union1d(idx_val, idx_test)
print(idx_unlabeled.shape)
# print(model.modified_adj)

