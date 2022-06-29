import torch
import torch_geometric
from .base_trainer import base_trainer
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

class gae_trainer(base_trainer):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device) -> None:
        super(gae_trainer, self).__init__(cfg, model, criterion, dataset_module, optimizer, device)

        data = dataset_module[0]
        self.train_data, self.val_data, self.test_data = data
    
    def train_one(self, device):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encode(self.train_data.x, self.train_data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=self.train_data.edge_index, num_nodes=self.train_data.num_nodes,
            num_neg_samples=self.train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [self.train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )

        edge_label = torch.cat([
            self.train_data.edge_label,
            self.train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = self.model.decode(z, edge_label_index).view(-1)

        loss = self.criterion(out, edge_label)
        loss.backward()
        self.optimizer.step()
        return loss 

    @torch.no_grad()
    def val_one(self, device, type="val"):
        self.model.eval()
        data = self.val_data
        if type == "test":
            data = self.test_data
        
        z = self.model.encode(data.x, data.edge_index)
        out = self.model.decode(z, data.edge_label_index).view(-1).sigmoid()
        return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

    @torch.no_grad()
    def test_one(self, device):
        return self.val_one(device, type="test")

