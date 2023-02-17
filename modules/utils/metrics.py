from itertools import count
import numpy as np
import torch
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.special import expit
import torch.nn.functional as F

K=20

"""Evaluation for static and discrete models: AUC, AP, Recall@K, NDCG@K"""
class Evaluation():
    def __init__(self, val_len, test_len) -> None:
        super().__init__()
        self.val_len = val_len
        self.test_len = test_len
        self.val_metrics = {"AUC": 0.0, "AP": 0.0, "recall": 0.0, "ndcg": 0.0}
        self.test_metrics = {"AUC": 0.0, "AP": 0.0, "recall": 0.0, "ndcg": 0.0}
        self.best_val_metrics = {"AUC": 0.0, "AP": 0.0, "recall": 0.0, "ndcg": 0.0}
        self.best_test_metrics = {"AUC": 0.0, "AP": 0.0, "recall": 0.0, "ndcg": 0.0}
        self.best_emb = None
        self.best_tr_embs = None

    def logging(self, emb, criterion = "AUC"):
        if self.val_metrics[criterion] > self.best_val_metrics[criterion]:
            self.best_test_metrics["AUC"] = self.test_metrics["AUC"]
            self.best_test_metrics["AP"] = self.test_metrics["AP"]
            self.best_val_metrics["AUC"] = self.val_metrics["AUC"]
            self.best_val_metrics["AP"] = self.val_metrics["AP"]

            self.best_test_metrics["recall"] = self.test_metrics["recall"]
            self.best_test_metrics["ndcg"] = self.test_metrics["ndcg"]
            self.best_val_metrics["recall"] = self.val_metrics["recall"]
            self.best_val_metrics["ndcg"] = self.val_metrics["ndcg"]

            self.best_emb = emb
    
    def decode(self, src, dst, z):
        dot = (z[src] * z[dst]).sum(dim=1)
        return torch.sigmoid(dot)

    def update(self, edges_pos, edges_neg, adj_orig_dense_list, embs):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        val_auc_scores, val_ap_scores = [], []
        test_auc_scores, test_ap_scores = [], []
        val_recall_scores, val_ndcg_scores = [], []
        test_recall_scores, test_ndcg_scores = [], []
        tscores = []
        fscores = []
        test_len = len(edges_pos)
        for i in range(test_len):
            # Predict on test set of edges
            emb = embs[i].cpu().detach() # n * d
            
            edges_p = edges_pos[i].cpu().detach().T
            edges_n = edges_neg[i].cpu().detach().T

            t_src, t_dst = edges_p
            f_src, f_dst = edges_n
            tscores = self.decode(t_src.long(), t_dst.long(), emb)
            fscores = self.decode(f_src, f_dst, emb)
 
            ntp = tscores.size(0)
            ntn = fscores.size(0)

            # Calculate AUC and AP
            score = torch.cat([tscores, fscores])
            labels = np.zeros(ntp + ntn, dtype=np.long)
            labels[:ntp] = 1

            ap = average_precision_score(labels, score)
            auc = roc_auc_score(labels, score)

            # Calculate Recall and NDCG
            emb_src = emb[t_src.numpy()] # emb_src: k=len(t_src) * d
            emb_node = emb # emb_node: n * d
            rank_scores = emb_src @ emb_node.T # rank_scores: k * n
            rank_scores.fill_diagonal_(0) # Change diagonal value to 0

            # # one hot preprocessing
            # t_one_hot = F.one_hot(t_dst.long(), num_classes=rank_scores.size(1)).to(torch.float32) # k * n
            # t_rank_scores = rank_scores * t_one_hot
            # f_one_hot = F.one_hot(f_dst.long(), num_classes=rank_scores.size(1)).to(torch.float32)
            # f_rank_scores = rank_scores * f_one_hot
            # rank_scores = t_rank_scores + f_rank_scores

            _, rank_indices = torch.sort(rank_scores, descending=True)
            rank_indices = rank_indices.cpu()
            gt = F.one_hot(t_dst.long(), num_classes=rank_scores.size(1)) # k * n
            gt = gt.numpy()
            
            binary_hit = []
            for j in range(len(t_src)):
                binary_hit.append(gt[j][rank_indices[j]])
            binary_hit = np.array(binary_hit, dtype=np.float32)

            recall = recall_at_k_batch(binary_hit, K)
            ndcg = ndcg_at_k_batch(binary_hit, K)
            total = sum([1 if (sum(gt_) > 0) else 0 for gt_ in gt])
            recall = sum(recall)/total
            ndcg = sum(ndcg)/total
            
            if i < self.val_len:
                # validation performance:
                val_auc_scores.append(auc)
                val_ap_scores.append(ap)
                val_recall_scores.append(recall)
                val_ndcg_scores.append(ndcg)
            else:
                # testing performance:
                test_auc_scores.append(auc)
                test_ap_scores.append(ap)
                test_recall_scores.append(recall)
                test_ndcg_scores.append(ndcg)

        self.val_metrics["AUC"] = np.mean(val_auc_scores)
        self.val_metrics["AP"] = np.mean(val_ap_scores)
        self.val_metrics["recall"] = np.mean(val_recall_scores)
        self.val_metrics["ndcg"] = np.mean(val_ndcg_scores)

        self.test_metrics["AUC"] = np.mean(test_auc_scores)
        self.test_metrics["AP"] = np.mean(test_ap_scores)
        self.test_metrics["recall"] = np.mean(test_recall_scores)
        self.test_metrics["ndcg"] = np.mean(test_ndcg_scores)

        self.logging(embs[-1].cpu().detach())

"""Evaluation for continuous models: AUC, AP"""
class ContEvaluation():
    def __init__(self, val_len, test_len) -> None:
        super().__init__()
        self.val_len = val_len
        self.test_len = test_len
        self.val_metrics = {"AUC": 0.0, "AP": 0.0, "recall": 0.0, "ndcg": 0.0}
        self.test_metrics = {"AUC": 0.0, "AP": 0.0, "recall": 0.0, "ndcg": 0.0}
        self.best_val_metrics = {"AUC": 0.0, "AP": 0.0, "recall": 0.0, "ndcg": 0.0}
        self.best_test_metrics = {"AUC": 0.0, "AP": 0.0, "recall": 0.0, "ndcg": 0.0}

    def logging(self, criterion = "AUC"):
        if self.val_metrics[criterion] > self.best_val_metrics[criterion]:
            self.best_test_metrics["AUC"] = self.test_metrics["AUC"]
            self.best_test_metrics["AP"] = self.test_metrics["AP"]
            self.best_val_metrics["AUC"] = self.val_metrics["AUC"]
            self.best_val_metrics["AP"] = self.val_metrics["AP"]
            
            self.best_test_metrics["recall"] = self.test_metrics["recall"]
            self.best_test_metrics["ndcg"] = self.test_metrics["ndcg"]
            self.best_val_metrics["recall"] = self.val_metrics["recall"]
            self.best_val_metrics["ndcg"] = self.val_metrics["ndcg"]
    
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

"""Ranking metrics functions"""
def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg

def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg

def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0] = np.inf

    res = (dcg / idcg)
    return res

def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num

def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / (hits.sum(axis=1) + 0.1))
    return res