


import torch
from torch import nn

from bbml.core.interfaces import Runnable, Serializable, Trainable
from bbml.core.datamodels.configs import FoundationConfig, TrainerConfig


class Foundation(Trainable, Runnable, Serializable, nn.Module):
    def __init__(self, config: FoundationConfig, train_config: TrainerConfig | None):
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.device = None
        self.dtype = None
    
    def to(self, *args, **kwargs):
        dtype = self.dtype
        device = self.device
        if args:
            arg = args[0]
            if isinstance(arg, torch.dtype):
                dtype = arg
            elif isinstance(arg, (str, torch.device, int)):
                device = arg
            elif isinstance(arg, torch.Tensor):
                dtype = arg.dtype
                device = arg.device
        if kwargs:    
            device = kwargs.get("device", device)
            dtype = kwargs.get("dtype", dtype)
        self.dtype = dtype
        self.device = device
        return super().to(*args, **kwargs)
        
