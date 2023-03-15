from typing_extensions import final
import networkx as nx
import os
import pickle
from matplotlib import pyplot as plt
from torch_geometric.utils import dense_to_sparse
import numpy as np
from torch import tensor
from scipy.sparse import csr_matrix
from matplotlib.pyplot import figure, text
import torch
from sklearn import preprocessing

def decode(src, dst, z):
    dot = (z[src] * z[dst]).sum(dim=1)
    return torch.sigmoid(dot)

def draw_graph_by_sparce_mx(adj_time_list, time_step, data_name, ptb_rate, attack_method, random_method, attacked=False, pred=False, tr=False):
    if not pred:
        csr_matrix_list = adj_time_list[time_step]
    else:
        csr_matrix_list = adj_time_list
    num_edges = np.sum(csr_matrix_list)//2
    tmp_list = dense_to_sparse(tensor(csr_matrix_list.toarray()))[0].cpu().tolist()
    fin_list = []
    for p in range(len(tmp_list[0])):
        fin_list.append((tmp_list[0][p],tmp_list[1][p]))
    Gtmp = nx.Graph()
    Gtmp.add_edges_from(fin_list)
    pos = nx.spring_layout(Gtmp)
    # print("connect components: ", list(nx.connected_components(Gtmp)))
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    d = dict(Gtmp.degree)
    ax.set_title("{} ptb_rate:{} time_step:{} attack_method:{} random_method:{} total_num_edges: {}".format(data_name, ptb_rate, time_step, attack_method, random_method, num_edges))
    nx.draw(Gtmp, node_size = 10, ax=ax, with_labels=True, pos=pos)
    _ = ax.axis('off')
    if attacked:
        file_name = "attacked_{}_{}_{}_{}_{}".format(data_name, ptb_rate, time_step, attack_method, random_method)
    else:
        file_name = "origin_{}_{}".format(data_name, time_step)
    
    if pred:
        file_name = "pred_{}_{}_{}_{}".format(data_name, attack_method, random_method, ptb_rate)
    if tr:
        file_name = "tr_{}_{}_{}_{}_{}".format(data_name, time_step, attack_method, random_method, ptb_rate)
    plt.savefig("/home/randomgraph/yichen14/code_base/visualization/{}.png".format(file_name))

DATASET_ROOT = "/home/randomgraph/data"
data_name = "enron10"
attack_method = "random"
random_method = "add"
ptb_rate = 0.0
test_len = 3
time_step = 0

# path = os.path.join(DATASET_ROOT, "attack_data", "{}_ptb_rate_{}_{}".format(data_name, ptb_rate, attack_method))
# pickle_path = os.path.join(path, "adj_ptb_{}_test_{}_{}.pickle".format(ptb_rate, test_len, random_method))
# with open(pickle_path, 'rb') as handle:
#     attacked_adj_time_list = pickle.load(handle,encoding="latin1")

adj_time_list_path = os.path.join(DATASET_ROOT, data_name, "adj_time_list.pickle")

with open(adj_time_list_path, 'rb') as handle:
    adj_time_list = pickle.load(handle,encoding="latin1")

# draw_graph_by_sparce_mx(attacked_adj_time_list, time_step, data_name, ptb_rate, attack_method, random_method, attacked=True)

# draw_graph_by_sparce_mx(adj_time_list, time_step, data_name, ptb_rate, attack_method, random_method, attacked=False)

def visualize_graph_from_emb(emb, time_step, attacked = False, pred = False, tr = False):
    # normalize emb
    emb = torch.nn.functional.normalize(emb, dim = 1) 
    predict = emb@emb.T
    final_pred =[]

    print(predict.min(), predict.max())

    # predict = torch.sigmoid(predict)
    # pred -= pred.min()
    # pred /= pred.max()
    # pred = (pred>0.5).float()
    print(predict.shape)
    print(predict.min(), predict.max())
    for i, row in enumerate(predict):
        row_list = []
        for j, num in enumerate(row):
            if num>=0.5 and i != j:
                row_list.append(1.0)
            else:
                row_list.append(0.0)
        final_pred.append(row_list)
    final_pred = csr_matrix(final_pred)
    # print(final_pred.shape)
    # draw prediciton snapshot
    draw_graph_by_sparce_mx(final_pred, time_step, data_name, ptb_rate, attack_method, random_method, attacked=attacked, pred=pred, tr=tr)
    
    return final_pred

path = os.path.join(DATASET_ROOT, "best_embedding", "static")
pickle_path = os.path.join(path, "{}_{}_{}.pickle".format(data_name, ptb_rate, attack_method))
# pickle_path = os.path.join(path, "enron10_Euler_original_emb.pickle")
with open(pickle_path, 'rb') as handle:
    emb = pickle.load(handle,encoding="latin1")

visualize_graph_from_emb(emb, 1, pred=True)

# visualize training snapshot
# pickle_path = os.path.join(path, "{}_tr_{}_{}.pickle".format(data_name, ptb_rate, attack_method))
# with open(pickle_path, 'rb') as handle:
#     embs = pickle.load(handle,encoding="latin1")

# print(embs.shape)

# for idx, emb in enumerate(embs):
#     visualize_graph_from_emb(emb, idx, tr=True, pred=True)


# # visualize original enron10 test snapshot
# draw_graph_by_sparce_mx(adj_time_list, time_step, data_name, ptb_rate, attack_method, random_method, attacked=False, pred=False)

