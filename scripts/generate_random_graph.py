from calendar import c
import __init_lib_path
import pickle
from deeprobust.graph.global_attack import Random
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
import os
import utils
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

CORA_PATH = os.path.join(utils.get_dataset_root(), 'Cora')

transform = T.Compose([
    T.ToUndirected(merge = True),
    T.ToDevice("cpu"),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                    add_negative_train_samples=False),
])  

cora_pyg = Planetoid(root = CORA_PATH, name="Cora", transform=transform)
#print(cora_pyg[0])
cora_dpr = Pyg2Dpr(cora_pyg[0])
print(cora_dpr)
adj, features, labels = cora_dpr.adj, cora_dpr.features, cora_dpr.labels
num_edges = cora_dpr.adj.shape[0]

print(cora_dpr.labels.shape)
model = Random()
model.attack(adj, n_perturbations=10)
modified_adj = model.modified_adj
cora_dpr.adj = modified_adj

pyg_data = Dpr2Pyg(cora_dpr)
print(pyg_data)
#print(modified_adj)
# raw_dir = '/home/ruijiew2/Documents/RandomGraph/data/static/'
# DATA_NAME = ['bill', 'election', 'timme']
# data_dict = {}
# for name in DATA_NAME:
#     with open(raw_dir + name+".pickle", 'rb') as f:
#         data = pickle.load(f)
#         data_dict[name] = data.toarray()

# for name in DATA_NAME:
