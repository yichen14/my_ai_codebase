import torch
import torch_geometric
from .base_trainer import base_trainer
from sklearn.metrics import roc_auc_score
import utils
from torch_geometric_temporal.signal import temporal_signal_split

class temp_graph_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, attack_func, device) -> None:
        super(temp_graph_trainer, self).__init__(cfg, model, criterion, dataset_module, optimizer, attack_func, device)

        # if(attack_func != None)
        #     attack_data = self.attack_func(dataset_module, cfg.ATTACK.ptb_rate, device)
        #     data = transform(attack_data.data)
        # else:
        #     data = transform(dataset_module[0])

        self.train_data, self.test_data = temporal_signal_split(dataset_module, train_ratio=0.2)

    def train_one(self, device):
        self.model.train()
        self.optimizer.zero_grad()
        loss = 0
        for time, snapshot in enumerate(self.train_data):
            y_hat = self.model(snapshot.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
            loss = loss + self.criterion(y_hat, snapshot.y.to(device))
        loss = loss / (time+1)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @torch.no_grad()
    def val_one(self, device):
        return self.test_one(device)

    @torch.no_grad()
    def test_one(self, device):
        self.model.eval()
        loss = 0
        for time, snapshot in enumerate(self.test_data):
            y_hat = self.model(snapshot.x.to(device), snapshot.edge_index.to(device), snapshot.edge_attr.to(device))
            loss = loss + self.criterion(y_hat, snapshot.y.to(device))
        loss = loss / (time+1)
        loss = loss.item()
        return loss
