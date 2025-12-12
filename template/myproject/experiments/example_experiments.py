"""Experiment stubs - register your experiments here.

Experiments orchestrate: data loading, model setup, training, evaluation.
Lifecycle: load() -> run() -> report() -> cleanup()

Two patterns:
  1. Training: override run() for single training loop
  2. Benchmarks/sweeps: override run_once(iteration) for repeated trials

Each experiment should represent a single concrete experiment, 
that means config values should be self-contained, rather than configured.

Registration: @ExperimentRegistry.register("name")
CLI: python scripts/run_experiment.py --name <name> -c configs/base.yaml
"""

import json
from typing import override
from pydantic import BaseModel
import torch
from torch.utils.data import Dataset
from bbml import (
    Foundation,
    FoundationConfig,
    Trainer,
    TrainerConfig,
    SimpleTrainer,
    DataPipe,
    Finetuner,
)
from bbml.utils.serialize import deep_serialize_pydantic

from myproject.experiments.base import Experiment
from myproject.experiments.registry import ExperimentRegistry


@ExperimentRegistry.register
class MyTrainingExperiment(Experiment):
    """
    Training experiment template. How training works in bbml.
    1. Create Foundation with configs
    2. Create DataPipe, add Dataset, add transforms
    3. Create SimpleTrainer with Foundation and DataPipe
    4. Train with trainer.train()
    5. Get samples and validation via trainer.test() and trainer.validate() 
    """
    def load(self):
        self.foundation: Foundation = Foundation(
            config=FoundationConfig(...),
            train_config=TrainerConfig(...),
        )
        self.foundation.to("cuda" if torch.cuda.is_available() else "cpu")
        
        datapipe = DataPipe(batch_size=self.trainer_config.batch_size, shuffle=True)
        datapipe.add_dataset(Dataset(...))
        datapipe.add_transforms(self.foundation.data_transforms)
        val_datapipe, train_pipe = datapipe.split(10, 90)
        
        self.trainer: Trainer = SimpleTrainer(
            model=self.foundation,
            train_config=self.trainer_config,
            train_datapipe=train_pipe,
            val_datapipe=val_datapipe,
        )

    @override
    def run(self) -> tuple:
        self.trainer.train()
        val_loss = self.trainer.validate()
        test_samples = self.trainer.test()
        return (val_loss, test_samples)
    

    def report(self, results: tuple):
        val_loss, test_samples = results
        serialized_samples = deep_serialize_pydantic(test_samples)
        serialized_results = {
            "val_loss": val_loss.item(),  # tensor to float
            "test_samples": serialized_samples,
        }

        results_dir = self.meta_config.report_dir / self.name
        results_dir.mkdir(exist_ok=True)
        with open(results_dir/f"report.json", "w") as f:
            json.dump(serialized_results, f, default=str)


@ExperimentRegistry.register
class MyFinetunedTrainingExperiment(MyTrainingExperiment):
    """
    Training experiment like above, but trained with a finetuner.
    """
    def load(self):
        super().load()
        self.finetuner: Finetuner = Finetuner(self.foundation)  # loads at initialization

    def cleanup(self):
        self.finetuner.remove()
        

@ExperimentRegistry.register
class MyAnalysisExperiment(Experiment):
    """
    Non-training experiment template (benchmarks, sweeps, analysis).

    Uses run_once() for iteration-based execution.
    cls.meta_config.iterations controls number of runs.
    """

    def load(self):
        self.foundation: Foundation = Foundation(
            config=FoundationConfig(...),
            train_config=None,
        )
        self.foundation.to("cuda" if torch.cuda.is_available() else "cpu")
        self.data: Dataset = Dataset(...)
        

    def run_once(self, iteration: int = 0) -> dict:
        inputs = self.foundation.input_model(**self.data[iteration])
        outputs = self.foundation.run(inputs)
        return (inputs, outputs)
        
        
    def report(self, results: list[dict]):
        serialized_results = deep_serialize_pydantic(results)

        results_dir = self.meta_config.report_dir / self.name
        results_dir.mkdir(exist_ok=True)
        with open(results_dir/"report.json", "w") as f:
            json.dump(serialized_results, f, default=str)
