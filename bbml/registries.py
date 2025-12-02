from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from bbml.core.logging import LoggingBackend
from bbml.core.registry import Registry


LoggingBackendRegistry: Registry[LoggingBackend] = Registry("LoggingBackend")
OptimizerRegistry: Registry[Optimizer] = Registry("Optimizer")
LRSchedulerRegistry: Registry[LRScheduler] = Registry("LRScheduler")