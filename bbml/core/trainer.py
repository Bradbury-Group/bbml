


from abc import ABC, abstractmethod
from typing import Any, Iterator

from torch import Tensor
from torch.utils.data import DataLoader
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

    def run_steps(
        self, loader: DataLoader, start_step: int, max_steps: int | None
    ) -> Iterator[tuple[int, int, Any]]:
        """Step-driven batch iterator for step-bounded training.

        Cycles ``loader`` across epochs (calling ``set_epoch`` on the loader's
        sampler and on the train datapipe at each wrap), never calls
        ``len(loader)``, and stops once ``max_steps`` is reached. Yields
        ``(epoch, batch_num, batch)``. Pure epoch mode iterates the loader
        directly instead.
        """
        step = start_step
        epoch = 0
        while max_steps is None or step < max_steps:
            self._set_loader_epoch(loader, epoch)
            produced = False
            for batch_num, batch in enumerate(loader):
                if max_steps is not None and step >= max_steps:
                    return
                yield epoch, batch_num, batch
                step += 1
                produced = True
            if not produced:
                raise RuntimeError(
                    "run_steps: loader yielded no batches; cannot cycle an empty loader."
                )
            epoch += 1

    def _set_loader_epoch(self, loader: DataLoader, epoch: int) -> None:
        """Set epoch on the loader's sampler and the train datapipe when present."""
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        pipe = getattr(self, "train_datapipe", None)
        if pipe is not None and hasattr(pipe, "set_epoch"):
            pipe.set_epoch(epoch)

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def validate(self) -> Tensor:
        ...
    
    @abstractmethod
    def test(self) -> Any:
        ...
    
