from deeprobust.graph.global_attack import Random
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
import os
import torch
import pickle
from tqdm import tqdm, trange

def random_attack_temporal(cfg, adj_matrix_lst, device):
    """
        perform a random attack on data
        Args:
            cfg - config object to get ptb_rate, test_len, and attack_data_path.
            adj_matrix_lst(scipy.sparse.csr_matrix) – Original (unperturbed) adjacency matrix.
        Return:
            attacked_csr_lst
    """
    attacked_csr_lst = []
    attacked_dense_lst = []
    attack_data_path = cfg.ATTACK.attack_data_path
    ptb_rate = cfg.ATTACK.ptb_rate
    test_len = cfg.DATASET.TEMPORAL.test_len
    random_type = cfg.ATTACK.RANDOM.method

    random_attack = Random(device=device)
    num_edges = adj_matrix_lst[0].shape[0]
    # num_modified = int(num_edges*ptb_rate)
    
    path = os.path.join(attack_data_path, "{}_ptb_rate_{}_random_{}".format(cfg.DATASET.dataset, ptb_rate, random_type))

    if "{}_ptb_rate_{}_random_{}".format(cfg.DATASET.dataset, ptb_rate, random_type) not in os.listdir(attack_data_path):
        # generate attacked data
        print("Random {} attack on dataset: {} ptb_rate: {}".format(random_type, cfg.DATASET.dataset, ptb_rate))
        os.mkdir(path)
        for time_step in trange(len(adj_matrix_lst)-1):
            adj_matrix = adj_matrix_lst[time_step]
            num_modified = int(ptb_rate * (adj_matrix.sum()//2))
            random_attack.attack(adj_matrix, n_perturbations = num_modified, type=random_type)
            pickle_path = os.path.join(path, "adj_ptb_{}_time_{}.pickle".format(ptb_rate, time_step))
            with open(pickle_path, 'ab') as handle:
                pickle.dump(random_attack.modified_adj, handle)

    # data already attacked 
    print("Load data from {}_ptb_rate_{}_random_{}".format(cfg.DATASET.dataset, ptb_rate, random_type), "test_len={}".format(test_len))
    for time_step in trange(len(adj_matrix_lst)):
        if time_step < (len(adj_matrix_lst)-test_len):
            pickle_path = os.path.join(path, "adj_ptb_{}_time_{}.pickle".format(ptb_rate, time_step))
            with open(pickle_path, 'rb') as handle:
                attacked_adj = pickle.load(handle,encoding="latin1")
                attacked_csr_lst.append(attacked_adj)
                attacked_dense_lst.append(torch.FloatTensor(attacked_adj.todense()))
        else:
            attacked_csr_lst.append(adj_matrix_lst[time_step])
            attacked_dense_lst.append(torch.FloatTensor(adj_matrix_lst[time_step].todense()))

    return attacked_csr_lst, attacked_dense_lst
