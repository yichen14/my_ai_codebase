from deeprobust.graph.global_attack import Random
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
import os
import torch
import pickle
from tqdm import tqdm, trange
import numpy as np
from utils import get_dataset_root
import logging

def random_attack_temporal(cfg, adj_matrix_lst, device):
    """
        perform a random attack on data
        Args:
            cfg - config object to get ptb_rate, test_len, and attack_data_path.
            adj_matrix_lst(scipy.sparse.csr_matrix) - Original (unperturbed) adjacency matrix.
        Return:
            attacked_csr_lst
    """
    attack_data_path = cfg.ATTACK.attack_data_path
    ptb_rate = cfg.ATTACK.ptb_rate
    test_len = cfg.DATASET.TEMPORAL.test_len

    random_attack = Random(device=device)
    path = os.path.join(get_dataset_root(), attack_data_path, "{}_ptb_rate_{}_random".format(cfg.DATASET.dataset, ptb_rate))

    if cfg.ATTACK.new_attack or not os.path.exists(os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))):
        # generate attacked data
        logging.info("Random attack on dataset: {} ptb_rate: {}".format(cfg.DATASET.dataset, ptb_rate))
        if not os.path.exists(path):
            os.mkdir(path)
        attack_data = []
        for time_step in range(len(adj_matrix_lst)-test_len):
            num_edges = np.sum(adj_matrix_lst[time_step])
            num_modified = int(num_edges*ptb_rate)//2
            adj_matrix = adj_matrix_lst[time_step]
            random_attack.attack(adj_matrix, n_perturbations = num_modified, type="add")
            attack_data.append(random_attack.modified_adj)
        pickle_path = os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))
        with open(pickle_path, 'ab') as handle:
            pickle.dump(attack_data, handle)

    # data already attacked
    logging.info("Load data from {}_ptb_rate_{}_random_{}.".format(cfg.DATASET.dataset, ptb_rate, test_len))
    pickle_path = os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))
    with open(pickle_path, 'rb') as handle:
        attacked_adj = pickle.load(handle,encoding="latin1")
        attacked_matrix_lst = attacked_adj
    
    assert len(attacked_matrix_lst) == len(adj_matrix_lst) - test_len
    
    for time_step in range(len(adj_matrix_lst) - test_len, len(adj_matrix_lst)):
            attacked_matrix_lst.append(adj_matrix_lst[time_step])

    assert len(attacked_matrix_lst) == len(adj_matrix_lst)
    return attacked_matrix_lst

