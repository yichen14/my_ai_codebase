import torch
from .base_trainer import base_trainer
from utils.metrics import Evaluation
import time
from torch.autograd import Variable
from tqdm import tqdm
import logging
import torch.nn as nn
from torch_geometric.utils import negative_sampling

class euler_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device) -> None:
        super(euler_trainer, self).__init__(cfg, model, criterion, dataset_module, optimizer, device)
        self.max_epochs = cfg.TRAIN.max_epochs
        self.log_epoch = cfg.TRAIN.log_epoch
        self.temporal_data = dataset_module
        self.cal_metric = Evaluation(self.cfg.DATASET.TEMPORAL.val_len, self.cfg.DATASET.TEMPORAL.test_len)

    def train(self):
        test_len = self.cfg.DATASET.TEMPORAL.test_len
        
        x_in = self.temporal_data.feat
        x_in = torch.stack(x_in).to(self.device)
        edge_idx_list = self.temporal_data.edge_idx_list
        adj_orig_dense_list = self.temporal_data.adj_orig_dense_list
        pos_edges_l, neg_edges_l = self.temporal_data.pos_edges_l, self.temporal_data.neg_edges_l

        train_start = 0
        train_end = self.temporal_data.time_step - test_len
        seq_end = self.temporal_data.time_step
        
        for i in range(self.temporal_data.time_step):
            adj_orig_dense_list[i] = adj_orig_dense_list[i].to(self.device) 
            pos_edges_l[i] = torch.tensor(pos_edges_l[i]).to(self.device)
            neg_edges_l[i] = torch.tensor(neg_edges_l[i]).to(self.device)
            edge_idx_list[i] = edge_idx_list[i].to(self.device)

        pbar = tqdm(range(self.max_epochs))
        start_time = time.time()

        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad()
            
            zs = self.model(x_in[train_start:train_end], edge_idx_list[train_start:train_end])
            
            neg_edges = []
            for i in range(train_start+1, train_end):
                neg_edges.append(negative_sampling(edge_idx_list[i], zs[i].size(0)).T)

            loss = self.model.loss_fn(edge_idx_list[train_start+1:train_end], neg_edges, zs[:-1])

            loss.backward()
            self.optimizer.step()

            nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            
            if epoch % self.log_epoch == 0:
                # edge_list_testing = [edge_idx_list[train_end] for i in range(test_len)]
                x_in_testing = torch.stack([x_in[train_end-1] for i in range(test_len)])
                edge_list_testing = [edge_idx_list[train_end-1] for i in range(test_len)]
                self.inference(x_in_testing, edge_list_testing, adj_orig_dense_list[train_end:seq_end], 
                             pos_edges_l[train_end:seq_end], neg_edges_l[train_end:seq_end])
                pbar.set_description('Epoch {}/{}, Loss {:.3f}, Test AUC {:.3f}, Test AP {:.3f}, Val AUC {:.3f}, Val AP {:.3f}, Time {:.1f}s'.format(epoch, self.max_epochs, loss.item(),self.cal_metric.test_metrics["AUC"], 
                    self.cal_metric.test_metrics["AP"], self.cal_metric.val_metrics["AUC"], self.cal_metric.val_metrics["AP"], time.time() - start_time))

        logging.info("Best performance: Test AUC {:.3f}, Test AP {:.3f}, Val AUC {:.3f}, Val AP {:.3f}".format(
                self.cal_metric.best_test_metrics["AUC"], self.cal_metric.best_test_metrics["AP"], self.cal_metric.best_val_metrics["AUC"], self.cal_metric.best_val_metrics["AP"]))

        return self.cal_metric.best_test_metrics["AUC"], self.cal_metric.best_test_metrics["AP"]

    @torch.no_grad()
    def inference(self, x_in, edge_list_testing, adj_orig_dense_list, pos_edges_l, neg_edges_l):
        self.model.eval()
        zs = self.model(x_in, edge_list_testing)
        self.cal_metric.update(pos_edges_l
                                , neg_edges_l
                                , adj_orig_dense_list
                                , zs)
