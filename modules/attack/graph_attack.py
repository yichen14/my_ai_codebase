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
            attacked_matrix_lst
    """
    attacked_matrix_lst = []
    attack_data_path = cfg.ATTACK.attack_data_path
    ptb_rate = cfg.ATTACK.ptb_rate
    test_len = cfg.DATASET.TEMPORAL.test_len
    random_attack = Random(device=device)
    num_edges = adj_matrix_lst[0].shape[0]
    num_modified = int(num_edges*ptb_rate)
    path = os.path.join(attack_data_path, "{}_ptb_rate_{}_random".format(cfg.DATASET.dataset, ptb_rate))

    if "{}_ptb_rate_{}_random".format(cfg.DATASET.dataset, ptb_rate) not in os.listdir(attack_data_path):
        # generate attacked data
        print("Random attack on dataset: {} ptb_rate: {}".format(cfg.DATASET.dataset, ptb_rate))
        os.mkdir(path)
        for time_step in trange(len(adj_matrix_lst)-1):
            adj_matrix = adj_matrix_lst[time_step]
            random_attack.attack(adj_matrix, n_perturbations = num_modified, type="flip")
            pickle_path = os.path.join(path, "adj_ptb_{}_time_{}.pickle".format(ptb_rate, time_step))
            with open(pickle_path, 'ab') as handle:
                pickle.dump(random_attack.modified_adj, handle)

    # data already attacked 
    print("Load data from {}_ptb_rate_{}_random.".format(cfg.DATASET.dataset, ptb_rate), "test_len=", test_len)
    for time_step in trange(len(adj_matrix_lst)):
        if time_step < (len(adj_matrix_lst)-test_len):
            pickle_path = os.path.join(path, "adj_ptb_{}_time_{}.pickle".format(ptb_rate, time_step))
            with open(pickle_path, 'rb') as handle:
                attacked_adj = pickle.load(handle,encoding="latin1")
                attacked_matrix_lst.append(attacked_adj)
        else:
            attacked_matrix_lst.append(adj_matrix_lst[time_step])

    return attacked_matrix_lst
