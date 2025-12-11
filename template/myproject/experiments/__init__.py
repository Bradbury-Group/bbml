"""Experiments module.

Stubs to extend:
  - MyTrainingExperiment: Foundation + DataPipe + SimpleTrainer
  - MyAnalysisExperiment: iteration-based benchmarks/sweeps

See example_experiments.py for templates.
See base.py for the Experiment lifecycle.
"""

from .base import Experiment, ExperimentConfig
from . import example_experiments  # noqa: F401 - registers experiments
from .registry import ExperimentRegistry

__all__ = ["Experiment", "ExperimentConfig", "ExperimentRegistry"]
