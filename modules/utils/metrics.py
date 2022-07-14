import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

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

    def update(self, edges_pos, edges_neg, adj_orig_dense_list, embs):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        val_auc_scores, val_ap_scores = [], []
        test_auc_scores, test_ap_scores = [], []
        
        for i in range(len(edges_pos)):
            # Predict on test set of edges
            emb = embs[i].cpu().detach().numpy()
            adj_rec = np.dot(emb, emb.T)
            adj_orig_t = adj_orig_dense_list[i]
            preds = []
            pos = []
            for e in edges_pos[i]:
                preds.append(sigmoid(adj_rec[e[0], e[1]]))
                pos.append(adj_orig_t[e[0], e[1]])
                
            preds_neg = []
            neg = []
            for e in edges_neg[i]:
                preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
                neg.append(adj_orig_t[e[0], e[1]])
            
            preds_all = np.hstack([preds, preds_neg])
            labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

            if i < self.val_len:
                # validation performance:
                val_auc_scores.append(roc_auc_score(labels_all, preds_all))
                val_ap_scores.append(average_precision_score(labels_all, preds_all))
            else:
                # testing performance:
                test_auc_scores.append(roc_auc_score(labels_all, preds_all))
                test_ap_scores.append(average_precision_score(labels_all, preds_all))

        self.val_metrics["AUC"] = np.mean(val_auc_scores)
        self.val_metrics["AP"] = np.mean(val_ap_scores)

        self.test_metrics["AUC"] = np.mean(test_auc_scores)
        self.test_metrics["AP"] = np.mean(test_ap_scores)

        self.logging()

