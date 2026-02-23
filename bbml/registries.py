from typing import TYPE_CHECKING
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from bbml.core.logging import LoggingBackend
from bbml.core.registry import Registry

if TYPE_CHECKING:
    from bbml.analysis.extractors.base import WeightExtractor
    from bbml.analysis.metrics.base import Metric


LoggingBackendRegistry: Registry[LoggingBackend] = Registry("LoggingBackend")
OptimizerRegistry: Registry[Optimizer] = Registry("Optimizer")
LRSchedulerRegistry: Registry[LRScheduler] = Registry("LRScheduler")
WeightExtractorRegistry: Registry = Registry("WeightExtractor")
MetricRegistry: Registry = Registry("Metric")