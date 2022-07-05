import torch
import torch_geometric
from .base_trainer import base_trainer
from sklearn.metrics import roc_auc_score
import utils
from torch_geometric.utils import negative_sampling

class autoencoder_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device) -> None:
        super(autoencoder_trainer, self).__init__(cfg, model, criterion, dataset_module, optimizer, device)

        data = dataset_module[0]
        self.train_data, self.val_data, self.test_data = data

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
        data = self.val_data
        if type == "test":
            data = self.test_data
        z = self.model.encode(data.x, data.edge_index)
        return self.model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

    @torch.no_grad()
    def test_one(self, device):
        return self.val_one(device, type="test")

