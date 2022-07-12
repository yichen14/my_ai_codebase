import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "robust graph learning on dynamic graphs")

    parser.add_argument('--device', type=int, default=0,
                        help='device id')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='fb',
                        help='Choose a dataset from {dblp, enron10, fb}')
    parser.add_argument('--data_dir', nargs='?', default='../data/',
                        help='Input data path.')

    parser.add_argument('--test_time_step', type=int, default=1, help='number of test time steps')
    parser.add_argument('--input_feat', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with BERT.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='recommendation batch size.')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='node embedding size.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=30,
                        help='Number of epoch for early stopping')

    parser.add_argument('--log_epoch', type=int, default=10,
                        help='Iter interval of printing loss.')
    parser.add_argument('--evaluate_epoch', type=int, default=10,
                        help='Epoch interval of evaluation.')

    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", default = "../configs/link_pred_temporal.yaml", type = str)
    parser.add_argument('--load', help="specify saved checkpoint to evaluate", required=False, type=str)
    parser.add_argument("--opts", help="Command line options to overwrite configs", default=[], nargs=argparse.REMAINDER)

    args = parser.parse_args()

    save_dir = '../trained_model/{}/entitydim{}_lr{}_pretrain{}/'.format(
        args.data_name, args.emb_dim, args.lr, args.input_feat)
    args.save_dir = save_dir
    return args