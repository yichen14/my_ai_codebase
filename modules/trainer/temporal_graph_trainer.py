
import torch
from .base_trainer import base_trainer
from utils.metrics import Evaluation
import time
from torch.autograd import Variable
from tqdm import tqdm
import logging
import torch.nn as nn

class temp_graph_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device) -> None:
        super(temp_graph_trainer, self).__init__(cfg, model, criterion, dataset_module, optimizer, device)
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
            
            kld_loss, nll_loss, _, _, hidden_st = self.model(x_in[train_start:train_end]
                                                , edge_idx_list[train_start:train_end]
                                                , adj_orig_dense_list[train_start:train_end])
            # loss = kld_loss + nll_loss
            loss = nll_loss + kld_loss
            nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            loss.backward()
            self.optimizer.step()
            
            if epoch % self.log_epoch == 0:
                # prepare testing input:
                edge_list_testing = [edge_idx_list[train_end - 1] for i in range(test_len)]
                x_in_testing = torch.stack([x_in[train_end - 1] for i in range(test_len)])
                
                self.inference(x_in_testing, edge_list_testing, adj_orig_dense_list[train_end:seq_end], 
                            hidden_st, pos_edges_l[train_end:seq_end], neg_edges_l[train_end:seq_end])
                pbar.set_description('Epoch {}/{}, Loss {:.3f}, Test AUC {:.3f}, Test AP {:.3f}, Val AUC {:.3f}, Val AP {:.3f}, Time {:.1f}s'.format(epoch, self.max_epochs, loss.item(),self.cal_metric.test_metrics["AUC"], 
                    self.cal_metric.test_metrics["AP"], self.cal_metric.val_metrics["AUC"], self.cal_metric.val_metrics["AP"], time.time() - start_time))

        logging.info("Best performance: Test AUC {:.3f}, Test AP {:.3f}, Val AUC {:.3f}, Val AP {:.3f}".format(
                self.cal_metric.best_test_metrics["AUC"], self.cal_metric.best_test_metrics["AP"], self.cal_metric.best_val_metrics["AUC"], self.cal_metric.best_val_metrics["AP"]))

        return self.cal_metric.best_test_metrics["AUC"], self.cal_metric.best_test_metrics["AP"]

    @torch.no_grad()
    def inference(self, x_in, edge_idx_list, adj_orig_dense_list, hidden_st, pos_edges_l, neg_edges_l):
        self.model.eval()
        _, _, enc_means, pri_means, _ = self.model(x_in
                                , edge_idx_list
                                , adj_orig_dense_list
                                , hidden_st)
        self.cal_metric.update(pos_edges_l
                                , neg_edges_l
                                , adj_orig_dense_list
                                , pri_means)
