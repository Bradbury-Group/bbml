


from abc import ABC, abstractmethod
from pathlib import Path

from torch.optim.optimizer import ParamsT
from bbml.core.interfaces import Serializable
from bbml.core.foundation import Foundation


class Finetuner(Serializable):
    def __init__(self, model: Foundation):
        self.model = model

        self.original_save = self.model.save
        self.model.save = self.save
        self.original_load = self.model.load
        self.model.load = self.load
        self.original_train_parameters = self.model.train_parameters
        self.model.train_parameters = self.train_parameters  # works since self.train_parameters is also a property
    
    def remove(self):
        self.model.save = self.original_save
        self.model.load = self.original_load
        self.model.train_parameters = self.original_train_parameters

    @property
    @abstractmethod
    def train_parameters(self) -> ParamsT:
        ...