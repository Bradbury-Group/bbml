"""Experiment lifecycle: load → run → report → cleanup."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetaExperimentConfig(BaseModel):
    """
    Meta config for experiment classes: shared config between experiments
    """
    report_dir: Path = Path("reports")
    seed: int = 42
    iterations: int = 1


class Experiment(ABC):
    """Example experiment.

    Lifecycle: load -> run → report -> cleanup

    For iteration-based experiments (benchmarks, sweeps), override run_once()
    and let run() handle the loop. For single-run experiments, override run().
    """

    def __init__(self, name: str):
        self.meta_config.report_dir.mkdir(parents=True, exist_ok=True)
        if name is None:
            raise ValueError("Please set a name for experiment")
        self.name = name

    meta_config: MetaExperimentConfig = MetaExperimentConfig()
    
    @classmethod
    def set_meta(cls, meta_config: MetaExperimentConfig):
        cls.meta_config = meta_config
    
    @classmethod
    def update_meta(cls, **kwargs):
        for k, v in kwargs.items():
            setattr(cls.meta_config, k, v) 

    def run(self) -> Any|list[Any]:
        """Main computation. Default: loop over run_once() for config.iterations."""
        results_list = []
        for i in range(self.meta_config.iterations):
            results = self.run_once(iteration=i)
            results_list.append(results)
        return results_list

    def run_all(self):
        """Execute: load -> run -> report -> cleanup."""
        self.load()
        run_results = self.run()
        self.report(run_results)
        self.cleanup()
    

    # Below are abstract methods to be implmemented
    @abstractmethod
    def load(self):
        """Load data, models, etc."""
        ...

    @abstractmethod
    def run_once(self, iteration: int = 0) -> Any:
        """Single iteration. Override for benchmarks/sweeps."""
        ...

    @abstractmethod
    def report(self, results: Any|list[Any]):
        """Save results."""
        ...

    @abstractmethod
    def cleanup(self):
        """Release resources."""
        ...
