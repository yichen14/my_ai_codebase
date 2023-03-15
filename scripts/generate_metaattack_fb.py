import os
import pickle
import numpy as np
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox
from deeprobust.graph.global_attack import Metattack
from scipy.sparse import csr_matrix

# def load_feat_and_label(data_name, data_path):
#     label = np.load(os.path.join(data_path, data_name, "label.npy"))
    
#     feat = np.load(os.path.join(data_path, data_name, "feat.npy"))
    
#     return label, feat

ptb_rate = 1.0
# nnode = 663
# attack_data_path = "attack_data"
# idx_train = np.arange(nnode//2)
# idx_val = np.arange(nnode//2, nnode//4*3)
# idx_test = np.arange(nnode//4*3, nnode)
# idx_unlabeled = np.union1d(idx_val, idx_test)
# label, feat = load_feat_and_label("fb", "/home/randomgraph/data")

# time_step = 5

# adj_time_list_path = os.path.join("/home/randomgraph/data", "fb", "adj_time_list.pickle")
# with open(adj_time_list_path, 'rb') as handle:
#     adj_matrix_lst = pickle.load(handle,encoding="latin1")

# adj = adj_matrix_lst[time_step].todense()

# num_edges = np.sum(adj_matrix_lst[time_step])
# num_modified = int(num_edges*ptb_rate)//2

# surrogate = GCN(nfeat=feat.shape[1], nclass=label.max().item()+1,
#     nhid=16, dropout=0, with_relu=False, with_bias=False, device="cuda:0").to("cuda:0")
# surrogate.fit(feat, adj, label, idx_train, idx_val, patience=30)
# model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=feat.shape,
#     attack_structure=True, attack_features=False, device="cuda:0", lambda_=0).to("cuda:0")

# model.attack(feat, adj, label, idx_train, idx_unlabeled, n_perturbations=num_modified, ll_constraint=False)
# attack_data = csr_matrix(np.array(model.modified_adj.cpu()))
attack_data = []
for time_step in range(0,6):
    pickle_path = os.path.join("/home/randomgraph/yichen14/code_base/data/cache", "adj_ptb_{}_fb_{}.pickle".format(ptb_rate, time_step))
    with open(pickle_path, 'rb') as handle:
        modified_matrix = pickle.load(handle,encoding="latin1")
    attack_data.append(modified_matrix)
print(len(attack_data))
assert len(attack_data) == 6
combined_path = os.path.join("/home/randomgraph/yichen14/code_base/data/cache", "adj_ptb_{}_test_{}.pickle".format(ptb_rate,3))
with open(combined_path, 'ab') as handle:
    pickle.dump(attack_data, handle)
    

    