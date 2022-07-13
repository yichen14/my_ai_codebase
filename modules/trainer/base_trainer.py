import math
import torch

from abc import abstractmethod

class base_trainer(object):
    def __init__(self, cfg, model, criterion, dataset_module, optimizer, device):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.device = device
        self.dataset_module = dataset_module
        self.optimizer = optimizer

    @abstractmethod
    def train_one(self, device, optimizer, epoch):
        raise NotImplementedError

    @abstractmethod
    def val_one(self, device):
        raise NotImplementedError

    @abstractmethod
    def test_one(self, device):
        raise NotImplementedError

    @abstractmethod
    def live_run(self, device):
        raise NotImplementedError
    
    def adapt_scheduler(self, start_epoch, scheduler):
        assert isinstance(start_epoch, int) and start_epoch >= 1
        if start_epoch > 1:
            if self.cfg.TRAIN.step_per_iter:
                elapsed_iter = math.ceil(len(self.train_set)  / self.cfg.TRAIN.batch_size) * (start_epoch - 1)
                for _ in range(elapsed_iter):
                    scheduler.step()
            else:
                # Tune LR scheduler
                for _ in range(1, start_epoch):
                    scheduler.step()

    def save_model(self, file_path):
        """Save default model (backbone_net, post_processor to a specified file path)

        Args:
            file_path (str): path to save the model
        """
        torch.save({
            "weights": self.model.state_dict(),
        }, file_path)
    
    def load_model(self, file_path):
        """Load weights for default model from a given file path

        Args:
            file_path (str): path to trained weights
        """
        trained_weight_dict = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(trained_weight_dict['weights'], strict=True)
