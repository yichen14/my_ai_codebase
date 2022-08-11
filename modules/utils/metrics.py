from itertools import count
import numpy as np
import torch
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.special import expit

"""Evaluation for static and discrete models"""
class Evaluation():
    def __init__(self, val_len, test_len) -> None:
        super().__init__()
        self.val_len = val_len
        self.test_len = test_len
        self.val_metrics = {"AUC": 0.0, "AP": 0.0}
        self.test_metrics = {"AUC": 0.0, "AP": 0.0}
        self.best_val_metrics = {"AUC": 0.0, "AP": 0.0}
        self.best_test_metrics = {"AUC": 0.0, "AP": 0.0}

    def logging(self, criterion = "AUC"):
        if self.val_metrics[criterion] > self.best_val_metrics[criterion]:
            self.best_test_metrics["AUC"] = self.test_metrics["AUC"]
            self.best_test_metrics["AP"] = self.test_metrics["AP"]
            self.best_val_metrics["AUC"] = self.val_metrics["AUC"]
            self.best_val_metrics["AP"] = self.val_metrics["AP"]
    
    def decode(self, src, dst, z):
        dot = (z[src] * z[dst]).sum(dim=1)
        return torch.sigmoid(dot)

    def update(self, edges_pos, edges_neg, adj_orig_dense_list, embs):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        val_auc_scores, val_ap_scores = [], []
        test_auc_scores, test_ap_scores = [], []
        tscores = []
        fscores = []
        test_len = len(edges_pos)
        for i in range(test_len):
            # Predict on test set of edges
            emb = embs[i].cpu().detach()
            
            edges_p = edges_pos[i].cpu().detach().T
            edges_n = edges_neg[i].cpu().detach().T

            t_src, t_dst = edges_p
            f_src, f_dst = edges_n
            tscores = self.decode(t_src.long(), t_dst.long(), emb)
            fscores = self.decode(f_src, f_dst, emb)
 

            ntp = tscores.size(0)
            ntn = fscores.size(0)

            score = torch.cat([tscores, fscores]).numpy()
            labels = np.zeros(ntp + ntn, dtype=np.long)
            labels[:ntp] = 1

            ap = average_precision_score(labels, score)
            auc = roc_auc_score(labels, score)
            
            if i < self.val_len:
                # validation performance:
                val_auc_scores.append(auc)
                val_ap_scores.append(ap)
            else:
                # testing performance:
                test_auc_scores.append(auc)
                test_ap_scores.append(ap)

        self.val_metrics["AUC"] = np.mean(val_auc_scores)
        self.val_metrics["AP"] = np.mean(val_ap_scores)

        self.test_metrics["AUC"] = np.mean(test_auc_scores)
        self.test_metrics["AP"] = np.mean(test_ap_scores)

        self.logging()

"""Evaluation for continuous models"""
class ContEvaluation():
    def __init__(self, val_len, test_len) -> None:
        super().__init__()
        self.val_len = val_len
        self.test_len = test_len
        self.val_metrics = {"AUC": 0.0, "AP": 0.0}
        self.test_metrics = {"AUC": 0.0, "AP": 0.0}
        self.best_val_metrics = {"AUC": 0.0, "AP": 0.0}
        self.best_test_metrics = {"AUC": 0.0, "AP": 0.0}

    def logging(self, criterion = "AUC"):
        if self.val_metrics[criterion] > self.best_val_metrics[criterion]:
            self.best_test_metrics["AUC"] = self.test_metrics["AUC"]
            self.best_test_metrics["AP"] = self.test_metrics["AP"]
            self.best_val_metrics["AUC"] = self.val_metrics["AUC"]
            self.best_val_metrics["AP"] = self.val_metrics["AP"]
    
    def eval_one_epoch(self, tgan, num_nerighbors, sampler, src, dst, ts):
        val_ap, val_auc = [], []
        with torch.no_grad():
            tgan = tgan.eval()
            TEST_BATCH_SIZE=30
            num_test_instance = len(src)
            num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
            for k in range(num_test_batch):
                s_idx = k * TEST_BATCH_SIZE
                e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
                if e_idx - s_idx == 1:
                    continue
                src_l_cut = src[s_idx:e_idx]
                dst_l_cut = dst[s_idx:e_idx]
                ts_l_cut = ts[s_idx:e_idx]

                size = len(src_l_cut)
                src_l_fake, dst_l_fake = sampler.sample(size)

                pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, num_nerighbors)
                
                pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                
                val_ap.append(average_precision_score(true_label, pred_score))
                val_auc.append(roc_auc_score(true_label, pred_score))
        return np.mean(val_ap), np.mean(val_auc)
    
    def update(self, tgan, num_neighbors, val_sampler, test_sampler,
                val_src, val_dst, val_ts, test_src, test_dst, test_ts):
        
        val_ap, val_auc = self.eval_one_epoch(tgan, num_neighbors, val_sampler, val_src, val_dst, val_ts)
        test_ap, test_auc = self.eval_one_epoch(tgan, num_neighbors, test_sampler, test_src, test_dst, test_ts)
        
        self.val_metrics["AUC"] = val_auc
        self.val_metrics["AP"] = val_ap
        self.test_metrics["AUC"] = test_auc
        self.test_metrics["AP"] = test_ap

        self.logging()
    
