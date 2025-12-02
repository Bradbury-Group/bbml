


from abc import ABC, abstractmethod
from bbml.core.interfaces import Trainable
from bbml.core.datapipe import DataPipe
from bbml.core.datamodels.configs import TrainerConfig
from bbml.core.foundation import Foundation


class Trainer(ABC):

    def __init__(
        self,
        model: Trainable | Foundation,
        train_config: TrainerConfig,
        datapipe: DataPipe,
    ):
        self.model = model
        self.train_config = train_config
        self.datapipe = datapipe
    
    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def validate(self):
        ...
    
    @abstractmethod
    def test(self):
        ...
    
