from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Experiment(ABC):
    def __init__(self):
        self.train_loader = None
        self.test_loader = None
        self.n_epochs = 1
        self.device = torch.device('cpu')
        self._is_configured = False

    @abstractmethod
    def _train_a_network(self, model) -> nn.Module:
        raise NotImplementedError()

    @abstractmethod
    def _evaluate_a_network(self, model) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def _configure(self):
        raise NotImplementedError()

    def _lazy_reconfigure(self):
        if not self._is_configured:
            self._configure()
            self._is_configured = True

    def reconfigure(self):
        self._is_configured = False
        self._configure()  # May throw exception
        self._is_configured = True

    def set_data_loaders(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader

    def set_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def set_device(self, device):
        self.device = device

    # TODO: implement various configuration options here
    # TODO: implement serialization / deserialization

    def train(self, model):
        self._lazy_reconfigure()
        network = None
        if isinstance(model, type):
            network = model()
        else:
            network = model
        # TODO: implement other interesting staff here including but not limited to
        if not isinstance(network, nn.Module):
            raise RuntimeError(f'Incorrect type of model {model}')
        network = network.to(self.device)
        return self._train_a_network(network)

    def evaluate(self, model):
        self._lazy_reconfigure()
        if not isinstance(model, nn.Module):
            raise RuntimeError(f'Incorrect type of model {model}')
        model = model.to(self.device)
        return self._evaluate_a_network(model)
