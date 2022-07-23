import os
import utils
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

def dispatcher(cfg, device = "cuda:0"):
    dataset_name = cfg.DATASET.dataset
    assert dataset_name != "none"
    if dataset_name == "cora":
        from torch_geometric.datasets import Planetoid
        CORA_PATH = os.path.join(utils.get_dataset_root(), 'Cora')  
        dataset_module = Planetoid(root = CORA_PATH, name="Cora")
        return dataset_module
    if dataset_name == "Chickenpox":
        from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
        loader = ChickenpoxDatasetLoader()
        return loader.get_dataset()
    if dataset_name in ["dblp", "enron10", "fb"]:
        from .temporal_graph import temporal_graph as temporal_graph_dataloader
        # from .egcn_graph import temporal_graph as temporal_graph_dataloader
        return temporal_graph_dataloader(cfg, device)
    else:
        raise NotImplementedError