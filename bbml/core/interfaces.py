from pathlib import Path
from typing import Any, TypeVar
from abc import ABC, abstractmethod

from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer, ParamsT
from pydantic import BaseModel

from bbml.core.data_transform import DataTransform


InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)


class Trainable(ABC):
    @abstractmethod
    def single_step(self, batch: dict[str, Any]) -> Tensor:
        ...

    @abstractmethod
    def get_train_parameters(self) -> ParamsT:
        ...

    @property
    @abstractmethod
    def data_transforms(self) -> list[str, DataTransform]:
        ...

    @property
    def optimizer(self) -> Optimizer | None:
        return None

    @property
    def lr_scheduler(self) -> LRScheduler | None:
        return None


class Runnable(ABC):
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


class Serializable(ABC):
    @abstractmethod
    def save(self, save_path: str | Path):
        ...

    @abstractmethod
    def load(self, load_path: str | Path):
        ...


