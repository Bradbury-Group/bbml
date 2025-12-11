from bbml.core.registry import Registry
from myproject.experiments.base import Experiment

ExperimentRegistry: Registry[Experiment] = Registry("Experiment")
