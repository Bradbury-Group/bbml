

from abc import ABC, abstractmethod
from typing import Any


class DataTransform(ABC):

    @abstractmethod
    def transform(self, input: Any) -> Any:
        ...
    
    @abstractmethod
    def batch_transform(self, input: list[Any]) -> Any:
        ...
    