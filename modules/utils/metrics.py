from itertools import count
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.special import expit

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

