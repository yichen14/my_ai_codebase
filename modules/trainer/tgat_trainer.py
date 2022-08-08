"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse
from tqdm import tqdm
import logging
import copy
import scipy
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from utils.tgat_utils import EarlyStopMonitor
from utils.metrics import ContEvaluation
from .base_trainer import base_trainer

"""
Modified from TGAT: https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs
"""

"""TGAT TRAINER"""

class tgat_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device) -> None:
        super(tgat_trainer, self).__init__(cfg, model, criterion, dataset_module, optimizer, device)
        self.max_epochs = cfg.TRAIN.max_epochs
        self.log_epoch = cfg.TRAIN.log_epoch
        self.batch_size = self.cfg.TRAIN.batch_size
        self.temporal_data = dataset_module
        self.cal_metric = ContEvaluation(self.cfg.DATASET.TEMPORAL.val_len, self.cfg.DATASET.TEMPORAL.test_len)
        self.criterion = torch.nn.BCELoss()
        self.num_neighbors = 16

        self.val_len = self.cfg.DATASET.TEMPORAL.val_len
        self.test_len = self.cfg.DATASET.TEMPORAL.test_len

        self.train_ngh_finder = self.temporal_data.train_ngh_finder
        self.full_ngh_finder = self.temporal_data.full_ngh_finder

        self.train_src_l = self.temporal_data.train_src_l
        self.train_dst_l = self.temporal_data.train_dst_l
        self.train_ts_l = self.temporal_data.train_ts_l
        self.train_e_idx_l = self.temporal_data.train_e_idx_l

        self.val_src_l = self.temporal_data.val_src_l
        self.val_dst_l = self.temporal_data.val_dst_l
        self.val_ts_l = self.temporal_data.val_ts_l
        self.val_e_idx_l = self.temporal_data.val_e_idx_l

        self.test_src_l = self.temporal_data.test_src_l
        self.test_dst_l = self.temporal_data.test_dst_l
        self.test_ts_l = self.temporal_data.test_ts_l
        self.test_e_idx_l = self.temporal_data.test_e_idx_l
        # print(self.test_src_l, self.test_dst_l, self.test_ts_l)
        # exit()


    def train(self):
        num_instance = len(self.temporal_data.train_src_l)
        num_batch = math.ceil(num_instance / self.batch_size)
        idx_list = np.arange(num_instance)
        np.random.shuffle(idx_list)
        
        pbar = tqdm(range(self.max_epochs))
        start_time = time.time()
        
        for epoch in pbar:
            self.model.ngh_finder = self.train_ngh_finder
            auc, ap, m_loss =[], [], []
            np.random.shuffle(idx_list)
            
            for k in range(num_batch):
                s_idx = k * self.batch_size
                e_idx = min(num_instance - 1, s_idx + self.batch_size)
                if e_idx - s_idx == 1:
                    continue

                src_l_cut, dst_l_cut = self.train_src_l[s_idx:e_idx], self.train_dst_l[s_idx:e_idx]
                ts_l_cut = self.train_ts_l[s_idx:e_idx]
                # label_l_cut = train_label_l[s_idx:e_idx]
                size = len(src_l_cut)
                src_l_fake, dst_l_fake = self.temporal_data.train_rand_sampler.sample(size)

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=self.device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=self.device)
                
                self.optimizer.zero_grad()
                self.model = self.model.train()
                pos_prob, neg_prob = self.model.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, self.num_neighbors)

                loss = self.criterion(pos_prob, pos_label)
                loss += self.criterion(neg_prob, neg_label)

                loss.backward()
                self.optimizer.step()

                # get training results
                # with torch.no_grad():
                #     tgan = tgan.eval()
                #     pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                #     true_label = np.concatenate([np.ones(size), np.zeros(size)])
                #     ap.append(average_precision_score(true_label, pred_score))
                #     m_loss.append(loss.item())
                #     auc.append(roc_auc_score(true_label, pred_score))

            if epoch % self.log_epoch == 0:
                self.model.ngh_finder = self.full_ngh_finder
                self.cal_metric.update(self.model, self.num_neighbors, self.temporal_data.val_rand_sampler, self.temporal_data.test_rand_sampler,
                                            self.val_src_l, self.val_dst_l, self.val_ts_l, self.test_src_l, self.test_dst_l, self.test_ts_l)
                pbar.set_description('Epoch {}/{}, Loss {:.3f}, Test AUC {:.3f}, Test AP {:.3f}, Val AUC {:.3f}, Val AP {:.3f}, Time {:.1f}s'.format(epoch, self.max_epochs, loss.item(),self.cal_metric.test_metrics["AUC"], 
                    self.cal_metric.test_metrics["AP"], self.cal_metric.val_metrics["AUC"], self.cal_metric.val_metrics["AP"], time.time() - start_time))

        logging.info("Best performance: Test AUC {:.3f}, Test AP {:.3f}, Val AUC {:.3f}, Val AP {:.3f}".format(
                self.cal_metric.best_test_metrics["AUC"], self.cal_metric.best_test_metrics["AP"], self.cal_metric.best_val_metrics["AUC"], self.cal_metric.best_val_metrics["AP"]))

        return self.cal_metric.best_test_metrics["AUC"], self.cal_metric.best_test_metrics["AP"]

