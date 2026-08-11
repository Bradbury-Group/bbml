from abc import abstractmethod
from pathlib import Path
from typing import Any

from torch.optim.optimizer import ParamsT

from bbml.core.foundation import Foundation
from bbml.core.interfaces import Serializable


class Finetuner(Serializable):
    def __init__(self, model: Foundation):
        self.model = model

        self.original_save = self.model.save
        self.model.save = self.save
        self.original_load = self.model.load
        self.model.load = self.load
        self.original_get_train_parameters = self.model.get_train_parameters
        self.model.get_train_parameters = self.get_train_parameters
        self.original_export_state_dict = getattr(self.model, "export_state_dict", None)
        if hasattr(self, "export_state_dict"): self.model.export_state_dict = self.export_state_dict

    def remove(self):
        self.model.save = self.original_save
        self.model.load = self.original_load
        self.model.get_train_parameters = self.original_get_train_parameters
        if self.original_export_state_dict is not None: self.model.export_state_dict = self.original_export_state_dict

    @abstractmethod
    def get_train_parameters(self) -> ParamsT:
        ...

    @abstractmethod
    def save(self, save_path: str | Path, **kwargs: Any):
        ...

    @abstractmethod
    def load(self, load_path: str | Path, **kwargs: Any):
        ...
