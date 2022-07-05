import __init_lib_path

import argparse
import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from config_guard import cfg, update_config_from_yaml
import torch_geometric.transforms as T
import utils
import dataset
import models
import trainer
import loss
from torch_geometric.datasets import Planetoid

import os
from tqdm import tqdm, trange

# Reference for link prediction: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py

CORA_PATH = os.path.join(utils.get_dataset_root(), 'Cora')

def parse_args():
    parser = argparse.ArgumentParser(description = "Eason's Deep Learning Codebase")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True, type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=False, type=str)
    parser.add_argument("--opts", help="Command line options to overwrite configs", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config_from_yaml(cfg, args)

    device = utils.guess_device()

    transform = T.Compose([
        T.ToUndirected(merge = True),
        T.ToDevice('cpu'),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
    ])  

    cora_dataset = Planetoid(root = CORA_PATH, name="Cora", transform=transform)
    
    num_features = cora_dataset.num_features
    if cfg.MODEL.encoder != "none":
        model_cls, encoder_cls = models.dispatcher(cfg)
        model = model_cls(encoder_cls(num_features))
    else:
        model_cls = models.dispatcher(cfg)
        model = model_cls(num_features).to(device)

    if cfg.TRAIN.OPTIMIZER.type == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr = cfg.TRAIN.initial_lr,
                                    weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr = cfg.TRAIN.initial_lr, momentum = cfg.TRAIN.OPTIMIZER.momentum,
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr = cfg.TRAIN.initial_lr, betas = (0.9, 0.999),
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    else:
        raise NotImplementedError("Got unsupported optimizer: {}".format(cfg.TRAIN.OPTIMIZER.type))
    criterion = loss.dispatcher(cfg)
    trainer_func = trainer.dispatcher(cfg)
    my_trainer = trainer_func(cfg, model, criterion, cora_dataset, optimizer, device)

    best_val_auc = final_test_auc = 0
    for epoch in range(1, cfg.TRAIN.max_epochs):
        loss_value = my_trainer.train_one(device)
        val_auc = my_trainer.val_one(device)
        test_auc = my_trainer.test_one(device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss_value:.4f}, Val: {val_auc:.4f}, '
            f'Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')

    #z = model.encode(test_data.x, test_data.edge_index)
    #final_edge_index = model.decode_all(z)

if __name__ == '__main__':
    main()