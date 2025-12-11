"""Experiments module.

Stubs to extend:
  - MyTrainingExperiment: Foundation + DataPipe + SimpleTrainer
  - MyAnalysisExperiment: iteration-based benchmarks/sweeps

See example_experiments.py for templates.
See base.py for the Experiment lifecycle.
"""

from myproject.experiments.base import Experiment, ExperimentConfig
from myproject.experiments import example_experiments  # noqa: F401 - registers experiments

__all__ = ["Experiment", "ExperimentConfig"]
