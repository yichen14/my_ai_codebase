import pickle
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import temporal_graph

class static_graph(temporal_graph):
    
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.raw_dir = '/home/ruijiew2/Documents/RandomGraph/data/static/'
        DATA_NAME = ['bill', 'election', 'timme']
        self.data_dict = {}
        for name in DATA_NAME:
            with open(self.raw_dir + name+".pickle", 'rb') as f:
                data = pickle.load(f)
                self.data_dict[name] = data.toarray()

        pass

    def get_A_matrix(self, name):
        """
        Return the adjacency matrix 
        Args:
            name: file name
        """
        return self.data_dict[name]

    @property
    def raw_file_names(self):
        return ['bill.pickle', 'election.pickle', 'timme.pickle']

    @property
    def processed_file_names(self):
        return ['bill.pt', 'election.pt', 'timme.pt']

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            #TODO: convert our Adjacency matrix data to torch_geometric.data.Data
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    


    