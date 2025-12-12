


from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from bbml.core.interfaces import Serializable, Trainable
from bbml.core.datapipe import DataPipe
from bbml.core.datamodels.configs import TrainerConfig
from bbml.core.foundation import Foundation


class Trainer(Serializable):

    def __init__(
        self,
        model: Trainable | Foundation,
        train_config: TrainerConfig,
        train_datapipe: DataPipe,
        val_datapipe: DataPipe | None,
        test_datapipe: DataPipe | None,
    ):
        self.model = model
        self.train_config = train_config
        self.train_datapipe = train_datapipe
        self.val_datapipe = val_datapipe
        self.test_datapipe = test_datapipe
    
    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def validate(self) -> Tensor:
        ...
    
    @abstractmethod
    def test(self) -> Any:
        ...
    
