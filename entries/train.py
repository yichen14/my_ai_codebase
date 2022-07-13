import __init_lib_path

import argparse
import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from config_guard import cfg, update_config_from_yaml

import utils
from utils.arg_parser import parse_args
from dataset.temporal_graph import temporal_graph
import dataset
import models
import trainer
import loss
import os
from tqdm import tqdm, trange

def setup(cfg, args):
    # get device
    device = args.device 
    # device = utils.guess_device()

    # set up dataset
    data = dataset.dispatcher(cfg, device)
    # data = temporal_graph(args.data_name, attack_flag = True, attack_func = attack_func)

    # set up model
    if cfg.MODEL.encoder != "none":
        model_cls, encoder_cls = models.dispatcher(cfg)
        model = model_cls(encoder_cls(cfg)).to(device)
    else:
        model_cls = models.dispatcher(cfg)
        model = model_cls(data.feat_dim, device).to(device)

    # set up optimizer
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
    # set up loss function (note that we do not need to give criterion to a graph autoencoder)
    criterion = loss.dispatcher(cfg)

    # set up trainer
    trainer_func = trainer.dispatcher(cfg)

    return data, model, trainer_func, optimizer, criterion, device

def main():
    args = parse_args()
    update_config_from_yaml(cfg, args)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    data, model, trainer_func, optimizer, criterion, device = setup(cfg, args)
    trainer = trainer_func(cfg, model, criterion, data, optimizer, device)

    trainer.train()
    # # start training
    # best_val_auc = final_test_auc = 0
    # for epoch in range(1, cfg.TRAIN.max_epochs):
    #     loss_value = my_trainer.train_one(device) # train
    #     val_auc = my_trainer.val_one(device) # eval
    #     test_auc = my_trainer.test_one(device) # test
    #     if val_auc > best_val_auc:
    #         best_val_auc = val_auc
    #         final_test_auc = test_auc
    #     print(f'Epoch: {epoch:03d}, Loss: {loss_value:.4f}, Val: {val_auc:.4f}, '
    #         f'Test: {test_auc:.4f}')

    # print(f'Final Test: {final_test_auc:.4f}')

    #z = model.encode(test_data.x, test_data.edge_index)
    #final_edge_index = model.decode_all(z)

if __name__ == '__main__':
    main()