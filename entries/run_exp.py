import __init_lib_path

import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from config_guard import cfg, update_config_from_yaml, update_cfg_from_args

import logging
import datetime

from utils.arg_parser import parse_args
import dataset
import loss

# models
from modules.models import egcn_h
from modules.models import egcn_o
from modules.models.tgat import TGAT

# trainer
from my_ai_codebase.modules.trainer import egcn_trainer

# tasker
from modules.tasker

def prepare_args(args):
    discrete_models = ["egcn_o", "egcn_h"]
    continuous_models = ["tgat"]

    if (
        not args.model
        in discrete_models + continuous_models
    ):
        raise NotImplementedError("Model {} not found".format(args.model))
    elif args.model in discrete_models:
        args.temporal_granularity = "discrete"


