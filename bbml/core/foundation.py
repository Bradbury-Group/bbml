

from pathlib import Path
from typing import Any, TypeVar
from abc import ABC, abstractmethod

from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer, ParamsT
from pydantic import BaseModel
import torch
from torch import nn

from bbml.core.data_transform import DataTransform
from bbml.core.interfaces import Runnable, Serializable, Trainable, InT, OutT
from bbml.core.datamodels.configs import FoundationConfig, TrainerConfig


class Foundation(Trainable, Runnable, Serializable, nn.Module):
    def __init__(self, config: FoundationConfig, train_config: TrainerConfig | None):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.device = None
        self.dtype = None
        

    # Below are copied abstract methods from .interfaces.py
    @abstractmethod
    def single_step(self, batch: dict[str, Any]) -> Tensor:
        ...

    @abstractmethod
    def get_train_parameters(self) -> ParamsT:
        ...

    @property
    @abstractmethod
    def data_transforms(self) -> dict[str, DataTransform]:
        ...

    @property
    def optimizer(self) -> Optimizer | None:
        return None

    @property
    def lr_scheduler(self) -> LRScheduler | None:
        return None

    @property
    @abstractmethod
    def input_model(self) -> type[InT]:
        ...

    @property
    @abstractmethod
    def output_model(self) -> type[OutT]:
        ...
    
    @abstractmethod
    def run(self, input: InT) -> OutT:
        ...

    @abstractmethod
    def save(self, save_path: str | Path):
        ...

    @abstractmethod
    def load(self, load_path: str | Path):
        ...

    
    # convience method for setting self.device and self.dtype, according to pytorch's Tensor.to and Module.to methods
    def to(self, *args, **kwargs):
        dtype = self.dtype
        device = self.device
        if args:
            arg = args[0]
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, (str, torch.device, int)):
                device = arg
            elif isinstance(arg, torch.Tensor):
                dtype = arg.dtype
                device = arg.device
        if kwargs:    
            device = kwargs.get("device", device)
            dtype = kwargs.get("dtype", dtype)
        self.dtype = dtype
        self.device = device
        return super().to(*args, **kwargs)