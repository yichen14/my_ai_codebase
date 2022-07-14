import __init_lib_path

import argparse
import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from config_guard import cfg, update_config_from_yaml, update_cfg_from_args

import utils
from utils.arg_parser import parse_args
from dataset.temporal_graph import temporal_graph
import dataset
import models
import trainer
import loss
from utils.metrics import Evaluation
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
    test_auc, test_ap = [], []
    for i in range(args.runs):
        args.seed += 5
        update_config_from_yaml(cfg, args)
        update_cfg_from_args(cfg, args)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    
        data, model, trainer_func, optimizer, criterion, device = setup(cfg, args)
        trainer = trainer_func(cfg, model, criterion, data, optimizer, device)

        test_auc_, test_ap_ = trainer.train()

        test_auc.append(test_auc_)
        test_ap.append(test_ap_)
    
    print("{} runs, Test AUC {:.3f} +- {:.3f}, Test AP {:.3f} +- {:.3f}".format(args.runs, np.mean(test_auc), np.std(test_auc), np.mean(test_ap), np.std(test_ap)))

if __name__ == '__main__':
    main()