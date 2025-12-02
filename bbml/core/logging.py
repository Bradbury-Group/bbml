
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping


class LoggingBackend(ABC):
    """
        Abstract base class for logging backends
        register with LoggingBackendRegistry
    """

    @abstractmethod
    def start(self, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def log(self, data: Mapping[str, Any], *, step: int | None = None, commit: bool = True):
        ...

    @abstractmethod
    def finish(self):
        ...



class AbstractLogger(ABC):
    @abstractmethod
    def start(self, service: str | Iterable[str], **kwargs: Any):
        ...

    @abstractmethod
    def log(self, data: Mapping[str, Any], *, step: int | None = None, commit: bool = True):
        ...

    @abstractmethod
    def finish(self):
        ...
