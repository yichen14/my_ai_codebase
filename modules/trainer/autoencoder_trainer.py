import torch
import torch_geometric
from .base_trainer import base_trainer
from sklearn.metrics import roc_auc_score
import utils
from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from utils.metrics import Evaluation
from tqdm import tqdm

class autoencoder_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device) -> None:
        super(autoencoder_trainer, self).__init__(cfg, model, criterion, dataset_module, optimizer, device)
        self.max_epochs = cfg.TRAIN.max_epochs
        self.log_epoch = cfg.TRAIN.log_epoch
        self.static_data = dataset_module
        self.cal_metric = Evaluation(self.cfg.DATASET.TEMPORAL.val_len, self.cfg.DATASET.TEMPORAL.test_len)

    def train(self):
        self.model.train()
        pbar = tqdm(range(self.max_epochs))
        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            z = self.model.encode(self.static_data.feat_static_train.to(self.device), self.static_data.edge_idx_train.to(self.device))
            loss = self.model.recon_loss(z, self.static_data.pos_edges_l_static_train.to(self.device),self.static_data.neg_edges_l_static_train.to(self.device))
            if self.cfg.MODEL.model == "VGAE":
                loss = loss + (1 / self.static_data.feat_static_train.shape[0]) * self.model.kl_loss()
            loss.backward()
            self.optimizer.step()      
            print(loss)
            if epoch % self.log_epoch == 0:
                print(self.val_one(self.device))

    def train_one(self, device):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        if self.cfg.MODEL.model == "VGAE":
            loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def val_one(self, device, type="val"):
        self.model.eval()

        z = self.model.encode(self.static_data.feat_static_val.to(self.device), self.static_data.edge_idx_val.to(self.device))

        return self.model.test(z, self.static_data.pos_edges_l_static_val.to(self.device),self.static_data.neg_edges_l_static_val.to(self.device))

    @torch.no_grad()
    def test_one(self, device):
        return self.val_one(device, type="test")

