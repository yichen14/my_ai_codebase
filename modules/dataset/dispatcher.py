import os
import utils
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

def dispatcher(cfg):
    dataset_name = cfg.DATASET.dataset
    assert dataset_name != "none"
    if dataset_name == "cora":
        from torch_geometric.datasets import Planetoid
        CORA_PATH = os.path.join(utils.get_dataset_root(), 'Cora')
        # transform = T.Compose([
        #     T.ToUndirected(merge = True),
        #     T.ToDevice('cpu'),
        #     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
        #                 split_labels=True, add_negative_train_samples=False),
        # ])      
        dataset_module = Planetoid(root = CORA_PATH, name="Cora")
        return dataset_module
    if dataset_name == "Chickenpox":
        from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
        loader = ChickenpoxDatasetLoader()
        return loader.get_dataset()
    else:
        raise NotImplementedError