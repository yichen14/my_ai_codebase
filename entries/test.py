import __init_lib_path

import argparse
import numpy as np
import torch

import torch_geometric.transforms as T
import utils
import dataset

transform = T.Compose([
    T.ToUndirected(merge = True),
    T.ToDevice('cuda'),
    T.RandomLinkSplit(num_val = 0.0005, num_test = 0.0001, is_undirected = True, add_negative_train_samples = False),
])  

def parse_args():
    parser = argparse.ArgumentParser(description = "Eason's Deep Learning Codebase")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True, type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=True, type=str)

def main():
    device = utils.guess_device()
    print(device)
    data= dataset.static_graph(device)
    A = data.get_A_matrix("election")
    
    pass

if __name__ == '__main__':
    main()