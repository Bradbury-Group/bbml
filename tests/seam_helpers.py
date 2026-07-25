"""Shared fixtures for the Foundation x Trainer seam tests."""
from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel
from torch import Tensor, nn

from bbml.core.datamodels.configs import FoundationConfig, TrainerConfig
from bbml.core.datapipe import DataPipe
from bbml.core.data_transform import DataTransform
from bbml.core.foundation import Foundation


class TinyInput(BaseModel):
    x: list = []


class TinyOutput(BaseModel):
    y: list = []


class TinyConfig(FoundationConfig):
    pass


class TinyFoundation(Foundation):
    """Minimal concrete Foundation: two 4->4 blocks + a head. single_step
    returns a scalar loss and a metrics dict exercising the validator."""

    def __init__(self, train_config: TrainerConfig | None = None, metrics: dict | None = None):
        super().__init__(TinyConfig(), train_config)
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.head = nn.Linear(4, 4)
        self.metrics = metrics if metrics is not None else {"aux": torch.tensor(1.5), "scalar": 2.0}
        self.train_start_calls: list[int] = []
        self.optimizer_step_calls: list[int] = []

    def single_step(self, batch: dict[str, Any]) -> tuple[Tensor, dict]:
        x = batch["x"]
        for blk in self.blocks:
            x = blk(x)
        loss = self.head(x).pow(2).mean()
        return loss, dict(self.metrics)

    def get_train_parameters(self):
        return self.parameters()

    @property
    def data_transforms(self) -> dict[str, DataTransform]:
        return {}

    @property
    def input_model(self):
        return TinyInput

    @property
    def output_model(self):
        return TinyOutput

    def run(self, input: TinyInput) -> TinyOutput:
        return TinyOutput()

    def fsdp_blocks(self):
        return self.blocks

    def on_train_start(self, step: int) -> None:
        self.train_start_calls.append(step)

    def on_optimizer_step(self, step: int) -> None:
        self.optimizer_step_calls.append(step)


class XTransform(DataTransform):
    """Stacks per-sample [4] float tensors into a [B, 4] batch."""

    def transform(self, inp):
        return torch.as_tensor(inp, dtype=torch.float32)

    def batch_transform(self, inp: list) -> Tensor:
        return torch.stack(inp)


class XDataset(torch.utils.data.Dataset):
    def __init__(self, n: int):
        self.rows = [{"x": [float(i)] * 4} for i in range(n)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_pipe(n: int = 8, batch_size: int = 2, shuffle: bool = False) -> DataPipe:
    pipe = DataPipe(batch_size=batch_size, shuffle=shuffle, num_workers=0)
    pipe.add_dataset(XDataset(n))
    pipe.add_transforms({"x": XTransform()})
    return pipe
