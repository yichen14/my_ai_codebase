import __init_lib_path

import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from config_guard import cfg, update_config_from_yaml, update_cfg_from_args

from utils.arg_parser import parse_args
import dataset
import models
import trainer
import loss
import logging
import datetime
import os

def setup(cfg, args):
    # get device
    if args.gpu:
        device = args.device
    else:
        device = torch.device('cpu')
    # device = utils.guess_device()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # set up dataset
    data = dataset.dispatcher(cfg, device)
    # data = temporal_graph(args.data_name, attack_flag = True, attack_func = attack_func)

    # set up model
    if cfg.MODEL.encoder != "none":
        model_cls, encoder_cls = models.dispatcher(cfg)
        model = model_cls(encoder_cls(cfg).to(device)).to(device)
    elif cfg.MODEL.model == "DYSAT":
        model_cls = models.dispatcher(cfg)
        test_len = cfg.DATASET.TEMPORAL.test_len
        model = model_cls(data.feat_dim, data.time_step - test_len).to(device)
    elif cfg.MODEL.model == "TGAT":
        model_cls = models.dispatcher(cfg)
        model = model_cls(data.train_ngh_finder, data.n_feat, data.e_feat).to(device)
    else:
        model_cls = models.dispatcher(cfg)
        if model_cls is not None:
            model = model_cls(data.feat_dim, device).to(device)
        else:
            model = None
    
    # set up optimizer
    if model is None:
        optimizer = None
    else:
        
        if cfg.TRAIN.OPTIMIZER.type == "adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr = cfg.TRAIN.initial_lr,
                                        weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
        elif cfg.TRAIN.OPTIMIZER.type == "SGD":
            optimizer = optim.SGD(model.parameters(), lr = cfg.TRAIN.initial_lr, momentum = cfg.TRAIN.OPTIMIZER.momentum,
                                    weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
        elif cfg.TRAIN.OPTIMIZER.type == "ADAM":
            optimizer = optim.Adam(model.parameters(), lr = cfg.TRAIN.initial_lr, betas = (0.9, 0.999),
                                    weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
            # optimizer = optim.Adam(model.parameters(), lr = cfg.TRAIN.initial_lr)                     
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
    update_cfg_from_args(cfg, args)
    test_auc, test_ap = [], []

    save_dir = './trained_model/{}_{}_{}_{}_lr_{}/'.format(
        cfg.DATASET.dataset, cfg.MODEL.model, cfg.ATTACK.method, cfg.ATTACK.ptb_rate, cfg.TRAIN.initial_lr)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.log_file = os.path.join(save_dir, 'log.txt')
    args.model_file = os.path.join(save_dir, 'model.pt')   

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()])
    logging.info("---------------------Experiment Info---------------------")
    logging.info("Dataset: {}, Learning Rate: {}, Model: {}, Attack Method: {}, ptb_rate: {}".format(cfg.DATASET.dataset, cfg.TRAIN.initial_lr, cfg.MODEL.model, cfg.ATTACK.method, cfg.ATTACK.ptb_rate))

    for i in range(1, args.runs+1):
        update_config_from_yaml(cfg, args)
        update_cfg_from_args(cfg, args)
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()])
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        logging.info("runs: {}/{}, time: {}, seed: {}".format(i, args.runs, datetime.datetime.now(), args.seed))
        logging.info("---------------------start training---------------------")

        data, model, trainer_func, optimizer, criterion, device = setup(cfg, args)
        trainer = trainer_func(cfg, model, criterion, data, optimizer, device)
        test_auc_, test_ap_ = trainer.train()
        
        logging.info("---------------------end training---------------------")
        test_auc.append(test_auc_)
        test_ap.append(test_ap_)

        args.seed += 5
        
    logging.info("{} runs, Test AUC {:.3f} +- {:.3f}, Test AP {:.3f} +- {:.3f}".format(args.runs, np.mean(test_auc), np.std(test_auc), np.mean(test_ap), np.std(test_ap)))

if __name__ == '__main__':
    main()