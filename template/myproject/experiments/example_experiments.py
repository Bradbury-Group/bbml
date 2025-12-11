"""Experiment stubs - register your experiments here.

Experiments orchestrate: data loading, model setup, training, evaluation.
Lifecycle: load() -> optimize() -> run() -> report() -> cleanup()

Two patterns:
  1. Training: override run() for single training loop
  2. Benchmarks/sweeps: override run_once(iteration) for repeated trials

Registration: @ExperimentRegistry.register("name")
CLI: python scripts/run_experiment.py --name <name> -c configs/base.yaml
"""

from bbml.core.datapipe import DataPipe
from bbml.core.datamodels.configs import TrainerConfig
from bbml.train.simple_trainer import SimpleTrainer

from myproject.experiments.base import Experiment, ExperimentConfig
from myproject.experiments.registry import ExperimentRegistry


@ExperimentRegistry.register("my_training")
class MyTrainingExperiment(Experiment):
    """Training experiment template.

    Wires up: Foundation + DataPipe + SimpleTrainer
    """

    def __init__(self, config: ExperimentConfig, trainer_config: TrainerConfig):
        """Receives configs from YAML via run_experiment.py.

        Args:
            config: ExperimentConfig (name, seed, report_dir, etc.)
            trainer_config: TrainerConfig (optimizer, scheduler, epochs, etc.)
        """
        super().__init__(config)
        self.trainer_config = trainer_config
        self.foundation = None
        self.trainer = None

    def load(self) -> None:
        """Set up model, data, trainer.

        Pattern:
          1. Create Foundation with configs
          2. Create DataPipe, add Dataset, add transforms
          3. Create SimpleTrainer with Foundation and DataPipe
        """
        # self.foundation = MyFoundation(
        #     config=MyFoundationConfig(**self.trainer_config.model_dump()),
        #     train_config=self.trainer_config,
        # )
        # self.foundation.to("cuda" if torch.cuda.is_available() else "cpu")
        #
        # train_pipe = DataPipe(batch_size=self.trainer_config.batch_size, shuffle=True)
        # train_pipe.add_dataset(MyDataset(split="train"))
        # train_pipe.add_transforms(self.foundation.data_transforms)
        #
        # self.trainer = SimpleTrainer(
        #     model=self.foundation,
        #     train_config=self.trainer_config,
        #     train_datapipe=train_pipe,
        #     val_datapipe=None,
        # )
        raise NotImplementedError

    def run(self) -> None:
        """Execute training."""
        # self.trainer.train()
        raise NotImplementedError


@ExperimentRegistry.register("my_analysis")
class MyAnalysisExperiment(Experiment):
    """
    Non-training experiment template (benchmarks, sweeps, analysis).

    Uses run_once() for iteration-based execution.
    config.iterations controls number of runs.
    """

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.data = None

    def load(self) -> None:
        """Load data or models for analysis."""
        raise NotImplementedError

    def run_once(self, iteration: int = 0) -> dict:
        """
        Single iteration. Return results dict.

        Called config.iterations times by base run().
        """
        raise NotImplementedError

    def report(self) -> None:
        """
        Save aggregated results.
        """
        raise NotImplementedError
