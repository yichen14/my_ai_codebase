import argparse
from xmlrpc.client import boolean
import yaml
import os

def load_yaml(file_path):
    """Load the YAML config file
    Args:
        file_path (_type_): _description_
    """
    with open(file_path, "r", errors="ignore") as stream:
        yaml_data = yaml.safe_load(stream)

    return yaml_data

def parse_args():
    parser = argparse.ArgumentParser(description = "robust graph learning on dynamic graphs")

    parser.add_argument('--gpu', type=int, default=1,
                        help='default is to use gpu')
    parser.add_argument('--device', type=int, default=0,
                        help='device id')

    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default=None,
                        help='Choose a dataset from {dblp, enron10, fb}')
    parser.add_argument('--data_dir', nargs='?', default=None,
                        help='Input data path.')
    
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name')                    

    parser.add_argument('--test_time_step', type=int, default=None, help='number of test time steps')

    parser.add_argument('--input_feat', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with BERT.')

    parser.add_argument('--batch_size', type=int, default=None,
                        help='recommendation batch size.')

    parser.add_argument('--emb_dim', type=int, default=None,
                        help='node embedding size.')

    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate.')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='Number of epoch.')
    
    parser.add_argument('--ptb_rate', type=float, default=None,
                        help='attack rate.')
    parser.add_argument('--attack_method', type=str, default=None,
                        help='attack_func.')

    parser.add_argument('--stopping_steps', type=int, default=None,
                        help='Number of epoch for early stopping')

    parser.add_argument('--log_interval', type=int, default=None,
                        help='Iter interval of printing loss.')
    parser.add_argument('--evaluate_epoch', type=int, default=None,
                        help='Epoch interval of evaluation.')

    parser.add_argument('--runs', type=int, default=10,
                        help='Independent runs with different seeds.')

    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", default = "./configs/link_pred_temporal.yaml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=False, type=str)
    parser.add_argument("--opts", help="Command line options to overwrite configs", default=[], nargs=argparse.REMAINDER)

    args = parser.parse_args()

    save_dir = './trained_model/{}_{}_{}_{}_lr_{}/'.format(
        args.data_name, args.model_name, args.attack_method, args.ptb_rate, args.lr)
    args.save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.log_file = os.path.join(save_dir, 'log.txt')
    args.model_file = os.path.join(save_dir, 'model.pt')

    return args