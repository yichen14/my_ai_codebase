from torch_geometric.datasets import Planetoid
import os
import utils

# CORA_PATH = os.path.join(utils.get_dataset_root(), 'Cora')

# download_ds = False if os.path.exists(CORA_PATH) else True

# cora_dataset = Planetoid(root = CORA_PATH, name="Cora", split="public")