import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility.

    Args:
        seed: Base seed for all RNGs.
        deterministic: Disable cudnn benchmark for full determinism (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def set_distributed_seed(seed: int, rank: int = 0, deterministic: bool = False):
    torch.manual_seed(seed)

    data_seed = seed + rank
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
