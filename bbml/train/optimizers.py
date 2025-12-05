from torch.optim import (
    Adadelta,
    Adagrad,
    Adam,
    AdamW,
    SparseAdam,
    Adamax,
    ASGD,
    SGD,
    RAdam,
    Rprop,
    RMSprop,
    NAdam,
    LBFGS,
)

from bbml.registries import OptimizerRegistry


# Register all PyTorch built-in optimizers
OptimizerRegistry.add("Adadelta", Adadelta)
OptimizerRegistry.add("Adagrad", Adagrad)
OptimizerRegistry.add("Adam", Adam)
OptimizerRegistry.add("AdamW", AdamW)
OptimizerRegistry.add("SparseAdam", SparseAdam)
OptimizerRegistry.add("Adamax", Adamax)
OptimizerRegistry.add("ASGD", ASGD)
OptimizerRegistry.add("SGD", SGD)
OptimizerRegistry.add("RAdam", RAdam)
OptimizerRegistry.add("Rprop", Rprop)
OptimizerRegistry.add("RMSprop", RMSprop)
OptimizerRegistry.add("NAdam", NAdam)
OptimizerRegistry.add("LBFGS", LBFGS)


try:
    import prodigyopt
    OptimizerRegistry.add("Prodigy", prodigyopt.Prodigy)
except ImportError:
    pass

try:
    import bitsandbytes as bnb
    OptimizerRegistry.add("AdamW8bit", bnb.optim.AdamW8bit)
except ImportError:
    pass