from xml.sax.handler import feature_external_ges
import os
import torch
import pickle
import random
from tqdm import tqdm, trange
import numpy as np
import logging
import random

from utils import get_dataset_root
from scipy.sparse import csr_matrix

from deeprobust.graph.global_attack import Random
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import MetaApprox
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from deeprobust.graph.global_attack import DICE


def load_feat_and_label(data_name, data_path):
    label = np.load(os.path.join(data_path, data_name, "label.npy"))
    
    feat = np.load(os.path.join(data_path, data_name, "feat.npy"))
    
    return label, feat

def node_emb_attack_temporal(cfg, adj_matrix_lst, device):
    """
        perform a node embedding attack on data
        Args:
            cfg - config object to get ptb_rate, test_len, and attack_data_path.
            adj_matrix_lst(scipy.sparse.csr_matrix) - Original (unperturbed) adjacency matrix.
        Return:
            attacked_csr_lst
    """
    attack_data_path = cfg.ATTACK.attack_data_path
    ptb_rate = cfg.ATTACK.ptb_rate
    test_len = cfg.DATASET.TEMPORAL.test_len
    feat_shape = cfg.TASK_SPECIFIC.GEOMETRIC.num_features

    if ptb_rate == 0.0:
        return adj_matrix_lst

    node_emb_attack = NodeEmbeddingAttack()
    path = os.path.join(get_dataset_root(), attack_data_path, "{}_ptb_rate_{}_node".format(cfg.DATASET.dataset, ptb_rate))
    if cfg.ATTACK.new_attack or not os.path.exists(os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))):
        # generate attacked data
        logging.info("Node embedding attack on dataset: {} ptb_rate: {}".format(cfg.DATASET.dataset, ptb_rate))
        if not os.path.exists(path):
            os.mkdir(path)
        attack_data = []
        for time_step in trange(len(adj_matrix_lst)-test_len):
            num_edges = np.sum(adj_matrix_lst[time_step])
            num_modified = int(num_edges*ptb_rate)//2
            adj_matrix = adj_matrix_lst[time_step]
            node_emb_attack.attack(adj_matrix, n_perturbations = num_modified, attack_type="add_by_remove", seed = cfg.seed, n_candidates=10000)
            attack_data.append(node_emb_attack.modified_adj)
        pickle_path = os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))
        with open(pickle_path, 'ab') as handle:
            pickle.dump(attack_data, handle)

    # data already attacked
    logging.info("Load data from {}_ptb_rate_{}_node_{}.".format(cfg.DATASET.dataset, ptb_rate, test_len))
    pickle_path = os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))
    with open(pickle_path, 'rb') as handle:
        attacked_adj = pickle.load(handle,encoding="latin1")
        attacked_matrix_lst = attacked_adj
    
    assert len(attacked_matrix_lst) == len(adj_matrix_lst) - test_len
    
    for time_step in range(len(adj_matrix_lst) - test_len, len(adj_matrix_lst)):
            attacked_matrix_lst.append(adj_matrix_lst[time_step])

    assert len(attacked_matrix_lst) == len(adj_matrix_lst)
    return attacked_matrix_lst


def dice_attack_temporal(cfg, adj_matrix_lst, device):
    """
        perform a DICE attack on data
        Args:
            cfg - config object to get ptb_rate, test_len, and attack_data_path.
            adj_matrix_lst(scipy.sparse.csr_matrix) - Original (unperturbed) adjacency matrix.
        Return:
            attacked_csr_lst
    """
    attack_data_path = cfg.ATTACK.attack_data_path
    ptb_rate = cfg.ATTACK.ptb_rate
    test_len = cfg.DATASET.TEMPORAL.test_len
    feat_shape = cfg.TASK_SPECIFIC.GEOMETRIC.num_features

    if ptb_rate == 0.0:
        return adj_matrix_lst
        
    dice_attack = DICE(device=device)
    path = os.path.join(get_dataset_root(), attack_data_path, "{}_ptb_rate_{}_dice".format(cfg.DATASET.dataset, ptb_rate))
    if cfg.ATTACK.new_attack or not os.path.exists(os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))):
        # generate attacked data
        logging.info("DICE attack on dataset: {} ptb_rate: {}".format(cfg.DATASET.dataset, ptb_rate))
        if not os.path.exists(path):
            os.mkdir(path)
        attack_data = []
        for time_step in trange(len(adj_matrix_lst)-test_len):
            num_edges = np.sum(adj_matrix_lst[time_step])
            num_modified = int(num_edges*ptb_rate)//2
            adj_matrix = adj_matrix_lst[time_step]
            dice_attack.attack(adj_matrix, labels=torch.zeros(feat_shape).long(), n_perturbations = num_modified)
            attack_data.append(dice_attack.modified_adj)
        pickle_path = os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))
        with open(pickle_path, 'ab') as handle:
            pickle.dump(attack_data, handle)

    # data already attacked
    logging.info("Load data from {}_ptb_rate_{}_dice_{}.".format(cfg.DATASET.dataset, ptb_rate, test_len))
    pickle_path = os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))
    with open(pickle_path, 'rb') as handle:
        attacked_adj = pickle.load(handle,encoding="latin1")
        attacked_matrix_lst = attacked_adj
    
    assert len(attacked_matrix_lst) == len(adj_matrix_lst) - test_len
    
    for time_step in range(len(adj_matrix_lst) - test_len, len(adj_matrix_lst)):
            attacked_matrix_lst.append(adj_matrix_lst[time_step])

    assert len(attacked_matrix_lst) == len(adj_matrix_lst)
    return attacked_matrix_lst

def meta_attack_temporal(cfg, adj_matrix_lst, device):
    attack_data_path = cfg.ATTACK.attack_data_path
    ptb_rate = cfg.ATTACK.ptb_rate
    test_len = cfg.DATASET.TEMPORAL.test_len
    nnode = cfg.TASK_SPECIFIC.GEOMETRIC.num_features
    data_name = cfg.DATASET.dataset

    if ptb_rate == 0.0:
        return adj_matrix_lst

    idx_train = np.arange(nnode//2)
    idx_val = np.arange(nnode//2, nnode//4*3)
    idx_test = np.arange(nnode//4*3, nnode)
    
    assert len(idx_test)+len(idx_train)+len(idx_val) == nnode

    idx_unlabeled = np.union1d(idx_val, idx_test)
    label, feat = load_feat_and_label(data_name, get_dataset_root())

    attacked_matrix_lst = []

    path = os.path.join(get_dataset_root(), attack_data_path, "{}_ptb_rate_{}_metaattack".format(cfg.DATASET.dataset, ptb_rate))
    if cfg.ATTACK.new_attack or not os.path.exists(os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))):
        # generate attacked data
        # print(os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len)))
        # print(cfg.ATTACK.new_attack)
        # exit()
        logging.info("Meta attack on dataset: {} ptb_rate: {}".format(cfg.DATASET.dataset, ptb_rate))
        if not os.path.exists(path):
            os.mkdir(path)
        attack_data = []

        for time_step in range(len(adj_matrix_lst)-test_len):
            torch.cuda.empty_cache()
            adj = adj_matrix_lst[time_step].todense()
            # adj = torch.tensor(adj)
            num_edges = np.sum(adj_matrix_lst[time_step])
            num_modified = int(num_edges*ptb_rate)//2

            surrogate = GCN(nfeat=feat.shape[1], nclass=label.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
            surrogate.fit(feat, adj, label, idx_train, idx_val, patience=30)
            model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=feat.shape,
                attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)

            model.attack(feat, adj, label, idx_train, idx_unlabeled, n_perturbations=num_modified, ll_constraint=False)
            attack_data.append(csr_matrix(np.array(model.modified_adj.cpu())))
            torch.cuda.empty_cache()
        pickle_path = os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))
        with open(pickle_path, 'ab') as handle:
            pickle.dump(attack_data, handle)

    # data already attacked
    logging.info("Load data from {}_ptb_rate_{}_meta_{}.".format(cfg.DATASET.dataset, ptb_rate, test_len))
    pickle_path = os.path.join(path, "adj_ptb_{}_test_{}.pickle".format(ptb_rate,test_len))
    with open(pickle_path, 'rb') as handle:
        attacked_adj = pickle.load(handle,encoding="latin1")
        attacked_matrix_lst = attacked_adj

    assert len(attacked_matrix_lst) == len(adj_matrix_lst) - test_len

    for time_step in range(len(adj_matrix_lst) - test_len, len(adj_matrix_lst)):
            attacked_matrix_lst.append(adj_matrix_lst[time_step])

    assert len(attacked_matrix_lst) == len(adj_matrix_lst)

    return attacked_matrix_lst

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

    random_method = "add" # set default random attack method here

    if ptb_rate == 0.0:
        return adj_matrix_lst
        
    random_attack = Random(device=device)
    path = os.path.join(get_dataset_root(), attack_data_path, "{}_ptb_rate_{}_random".format(cfg.DATASET.dataset, ptb_rate))
    if cfg.ATTACK.new_attack or not os.path.exists(os.path.join(path, "adj_ptb_{}_test_{}_{}.pickle".format(ptb_rate, test_len, random_method))):
        # generate attacked data
        logging.info("Random attack on dataset: {} ptb_rate: {} method: {}".format(cfg.DATASET.dataset, ptb_rate, random_method))
        if not os.path.exists(path):
            os.mkdir(path)
        attack_data = []
        for time_step in range(len(adj_matrix_lst)-test_len):
            num_edges = np.sum(adj_matrix_lst[time_step])
            num_modified = int(num_edges*ptb_rate)//2
            adj_matrix = adj_matrix_lst[time_step]
            random_attack.attack(adj_matrix, n_perturbations = num_modified, type=random_method)
            attack_data.append(random_attack.modified_adj)
        pickle_path = os.path.join(path, "adj_ptb_{}_test_{}_{}.pickle".format(ptb_rate, test_len, random_method))
        with open(pickle_path, 'ab') as handle:
            pickle.dump(attack_data, handle)

    # data already attacked
    logging.info("Load data from {}_ptb_rate_{}_random_{}_{}.".format(cfg.DATASET.dataset, ptb_rate, test_len, random_method))
    pickle_path = os.path.join(path, "adj_ptb_{}_test_{}_{}.pickle".format(ptb_rate,test_len, random_method))
    with open(pickle_path, 'rb') as handle:
        attacked_adj = pickle.load(handle,encoding="latin1")
        attacked_matrix_lst = attacked_adj
    
    assert len(attacked_matrix_lst) == len(adj_matrix_lst) - test_len
    
    for time_step in range(len(adj_matrix_lst) - test_len, len(adj_matrix_lst)):
            attacked_matrix_lst.append(adj_matrix_lst[time_step])

    assert len(attacked_matrix_lst) == len(adj_matrix_lst)
    return attacked_matrix_lst

def temporal_shift_attack(cfg, adj_matrix_lst, device):
    attack_data_path = cfg.ATTACK.attack_data_path
    ptb_rate = cfg.ATTACK.ptb_rate
    test_len = cfg.DATASET.TEMPORAL.test_len

    if ptb_rate == 0.0:
        return adj_matrix_lst

    N = len(adj_matrix_lst)

    modified_time_step = int((len(adj_matrix_lst)-test_len) * ptb_rate)

    attacked_matrix_lst = []
    # attacked_matrix_lst.append(adj_matrix_lst[:modified_time_step])
    for _ in range(modified_time_step):
        idx = random.randint(0,len(adj_matrix_lst)-1-test_len)
        attacked_matrix_lst.append(adj_matrix_lst.pop(idx))
    
    attacked_matrix_lst+=adj_matrix_lst

    assert len(attacked_matrix_lst) == N

    return attacked_matrix_lst

