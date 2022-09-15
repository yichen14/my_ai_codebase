import time
from tqdm import tqdm
import logging
import networkx as nx
from collections import defaultdict
import copy
import scipy
from scipy.sparse import csr_matrix
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_trainer import base_trainer
from utils.metrics import Evaluation
from utils.minibatch import MyDataset
from utils.random_walk import Graph_RandomWalk
from utils import get_dataset_root

"""
Parts of this code file are derive from 
DySAT PyTorch: https://github.com/FeiGSSS/DySAT_pytorch
"""

"""Random walk-based pair generation."""
def run_random_walks_n2v(graph, adj, num_walks, walk_len):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using 
        the sampling strategy of node2vec (deepwalk)"""
    nx_G = nx.Graph()
    # TypeError: 'coo_matrix' object is not subscriptable
    adj = adj.tocsr()
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])
    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 10
    pairs = defaultdict(list)
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    # print("# nodes with random walk samples: {}".format(len(pairs)))
    # print("# sampled pairs: {}".format(pairs_cnt))
    return pairs

def get_context_pairs(graphs, adjs):
        """ Load/generate context pairs for each snapshot through random walk sampling."""
        print("Computing training pairs ...")
        context_pairs_train = []
        for i in range(len(graphs)):
            context_pairs_train.append(run_random_walks_n2v(graphs[i], adjs[i], num_walks=10, walk_len=5))

        return context_pairs_train

"""feed_dict to device"""
def to_device(batch, device):
    feed_dict = copy.deepcopy(batch)
    node_1, node_2, node_2_negative, graphs = feed_dict.values()
    # to device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]
    feed_dict["graphs"] = [g.to(device) for g in graphs]

    return feed_dict

"""Split feed_dict"""
def split_feed_dict(batch, train_len, test_len):
    feed_dict = copy.deepcopy(batch)
    node_1, node_2, node_2_negative, graphs = feed_dict.values()
    feed_dict_train, feed_dict_test = {}, {}
    
    feed_dict_train['node_1']=node_1[:train_len]
    feed_dict_train['node_2']=node_2[:train_len]
    feed_dict_train['node_2_neg']=node_2_negative[:train_len]
    feed_dict_train["graphs"] = graphs[:train_len]

    feed_dict_test['node_1']=node_1[len(node_1)-test_len:]
    feed_dict_test['node_2']=node_2[len(node_1)-test_len:]
    feed_dict_test['node_2_neg']=node_2_negative[len(node_1)-test_len:]
    feed_dict_test["graphs"] = graphs[len(node_1)-test_len:]

    return feed_dict_train, feed_dict_test

"""DYSAT TRAINER"""
class dysat_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device) -> None:
        super(dysat_trainer, self).__init__(cfg, model, criterion, dataset_module, optimizer, device)
        self.max_epochs = cfg.TRAIN.max_epochs
        self.log_epoch = cfg.TRAIN.log_epoch
        self.temporal_data = dataset_module
        self.cal_metric = Evaluation(self.cfg.DATASET.TEMPORAL.val_len, self.cfg.DATASET.TEMPORAL.test_len)
        self.model.to(self.device)
        self.test_len = self.cfg.DATASET.TEMPORAL.test_len
        self.batch_size = self.cfg.TRAIN.batch_size

    def build_dataloader(self):
        graphs = self.temporal_data.graphs
        adj_time_list = self.temporal_data.adj_time_list
        time_steps = self.temporal_data.time_step

        feats = [scipy.sparse.identity(self.temporal_data.num_nodes).tocsr() for i in range(time_steps)]

        context_pairs_train = get_context_pairs(graphs, adj_time_list)

        # build dataloader
        dataset = MyDataset(graphs, feats, adj_time_list, context_pairs_train, time_steps)
        dataloader = DataLoader(dataset,
                                # batch_size=self.temporal_data.num_nodes,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=10,
                                collate_fn=MyDataset.collate_fn)
        
        return dataloader

    def train(self):
        adj_orig_dense_list = self.temporal_data.adj_orig_dense_list
        pos_edges_l, neg_edges_l = self.temporal_data.pos_edges_l, self.temporal_data.neg_edges_l

        dataloader = self.build_dataloader()

        train_start = 0
        train_end = self.temporal_data.time_step - self.test_len
        seq_end = self.temporal_data.time_step
        
        for i in range(self.temporal_data.time_step):
            adj_orig_dense_list[i] = adj_orig_dense_list[i].to(self.device) 
            pos_edges_l[i] = torch.tensor(pos_edges_l[i]).to(self.device)
            neg_edges_l[i] = torch.tensor(neg_edges_l[i]).to(self.device)

        pbar = tqdm(range(1, self.max_epochs+1))
        start_time = time.time()

        for epoch in pbar:
            self.model.train()
            for idx, feed_dict in enumerate(dataloader):
                feed_dict = to_device(feed_dict, self.device)
                feed_dict_train, feed_dict_test = split_feed_dict(feed_dict, train_end-train_start, self.test_len)
                self.optimizer.zero_grad()
                loss = self.model.get_loss(feed_dict_train)
                loss.backward()
                self.optimizer.step()
            
            # self.model.eval()
            # emb = self.model(feed_dict["graphs"])
            # print(emb.size()) # enron10, [184, 11, 128]
            
            # nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            
            if epoch % self.log_epoch == 0:
                # prepare testing input:
                graphs_testing = [feed_dict["graphs"][train_end-1] for _ in range(train_end-train_start)] # Repeat graph (single-step prediction)
                # graphs_testing = feed_dict["graphs"] # Repeat emb (multi-step prediction)
                
                self.inference(graphs_testing, adj_orig_dense_list[train_end:seq_end], 
                             pos_edges_l[train_end:seq_end], neg_edges_l[train_end:seq_end])
                pbar.set_description('Epoch {}/{}, Loss {:.3f}, Test AUC {:.3f}, Test AP {:.3f}, Val AUC {:.3f}, Val AP {:.3f}, Time {:.1f}s'.format(epoch, self.max_epochs, loss.item(),self.cal_metric.test_metrics["AUC"], 
                    self.cal_metric.test_metrics["AP"], self.cal_metric.val_metrics["AUC"], self.cal_metric.val_metrics["AP"], time.time() - start_time))

        logging.info("Best performance: Test AUC {:.3f}, Test AP {:.3f}, Val AUC {:.3f}, Val AP {:.3f}".format(
                self.cal_metric.best_test_metrics["AUC"], self.cal_metric.best_test_metrics["AP"], self.cal_metric.best_val_metrics["AUC"], self.cal_metric.best_val_metrics["AP"]))

        return self.cal_metric.best_test_metrics["AUC"], self.cal_metric.best_test_metrics["AP"]

    @torch.no_grad()
    def inference(self, graphs, adj_orig_dense_list, pos_edges_l, neg_edges_l):
        self.model.eval()
        # Repeat emb (multi-step prediction)
        # emb = self.model(graphs)[:,-1,:] # The last snapshot
        # embs = [emb for _ in range(self.test_len)]
        
        # Repeat graph (single-step prediction)
        emb = self.model(graphs)
        embs = [emb[:,i,:] for i in range(self.test_len)]
        self.cal_metric.update(pos_edges_l
                                , neg_edges_l
                                , adj_orig_dense_list
                                , embs)
