import torch
import os
import urllib

DATASET_ROOT = "/home/randomgraph/data"

def set_dataset_root(path):
    global DATASET_ROOT
    DATASET_ROOT = path

def download_file(url, local_path):
    g = urllib.request.urlopen(url)
    with open(local_path, 'b+w') as f:
        f.write(g.read())

def get_dataset_root():
    global DATASET_ROOT
    try:
        return os.environ['DATASET_ROOT']
    except KeyError:
        return DATASET_ROOT

def guess_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')