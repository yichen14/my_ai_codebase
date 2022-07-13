import torch 
import torch_geometric
from .base_trainer import base_trainer
from sklearn.metrics import roc_auc_score
from utils.metrics import get_roc_scores
from utils.generators import link_prediction
import time
from torch.autograd import Variable
from torch import nn
import numpy as np

class egcn_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device):
        super().__init__(cfg, model, criterion, dataset_module, optimizer, device)
        self.max_epochs = cfg.TRAIN.max_epochs
        self.log_epoch = cfg.TRAIN.log_epoch
        self.temporal_data = dataset_module
    
    def train(self, test_len = 1):

        x_in = self.temporal_data.feat
        x_in = Variable(torch.stack(x_in)).to(self.device)
        edge_idx_list = self.temporal_data.edge_idx_list
        adj_orig_dense_list = self.temporal_data.adj_orig_dense_list
        pos_edges_l, neg_edges_l = self.temporal_data.pos_edges_l, self.temporal_data.neg_edges_l

        seq_start = 0
        seq_end = self.temporal_data.time_step - test_len
        seq_len = self.temporal_data.time_step

        print(len(edge_idx_list))
        print(len(adj_orig_dense_list))

        for i in range(self.temporal_data.time_step):
            pos_edges_l[i] = torch.tensor(pos_edges_l[i]).to(self.device)
            neg_edges_l[i] = torch.tensor(neg_edges_l[i]).to(self.device)
            # adj_orig_dense_list[i] = torch.tensor(adj_orig_dense_list[i]).to(self.device)
            edge_idx_list[i] = torch.tensor(edge_idx_list[i]).to(self.device)

        for k in range(1, self.max_epochs+1):
            self.model.train()
            self.optimizer.zero_grad()

            # Get embedding 
            zs = self.model(adj_orig_dense_list, x_in)

            p, n, z = link_prediction()

            
            

            

            print('epoch: ', k)
            print('kld_loss =', kld_loss.mean().item())
            print('nll_loss =', nll_loss.mean().item())
            print('loss =', loss.mean().item())

            if k % self.log_epoch == 0:
                _, _, enc_means, pri_means, _ = self.model(x_in[seq_end:seq_len]
                                              , edge_idx_list[seq_end:seq_len]
                                              , adj_orig_dense_list[seq_end:seq_len]
                                              , hidden_st)
        
                auc_scores_prd, ap_scores_prd = get_roc_scores(pos_edges_l[seq_end:seq_len]
                                                        , neg_edges_l[seq_end:seq_len]
                                                        , adj_orig_dense_list[seq_end:seq_len]
                                                        , pri_means)
        
                # auc_scores_prd_new, ap_scores_prd_new = get_roc_scores(pos_edges_l_n[seq_end:seq_len]
                #                                                 , false_edges_l_n[seq_end:seq_len]
                #                                                 , adj_orig_dense_list[seq_end:seq_len]
                #                                                 , pri_means)
        
    
                print('----------------------------------')
                print('epoch: ', k)
                print('Link Prediction')
                print('link_prd_auc_mean', np.mean(np.array(auc_scores_prd)))
                print('link_prd_ap_mean', np.mean(np.array(ap_scores_prd)))
                print('----------------------------------')
                # print('New Link Prediction')
                # print('new_link_prd_auc_mean', np.mean(np.array(auc_scores_prd_new)))
                # print('new_link_prd_ap_mean', np.mean(np.array(ap_scores_prd_new)))
                # print('----------------------------------')
        print('----------------------------------')