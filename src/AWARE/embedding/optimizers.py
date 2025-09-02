import torch
from AWARE.utils import logger
registry = {
    "adam": torch.optim.Adam,
    "nadam": torch.optim.NAdam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "adamax": torch.optim.Adamax,
    "adamw": torch.optim.AdamW,
    "sparse_adam": torch.optim.SparseAdam,
    "lbfgs": torch.optim.LBFGS,
}

def get_optimizer(name: str, params: list[torch.Tensor] | torch.Tensor, **kwargs) -> torch.optim.Optimizer:
    if name not in registry:
        raise ValueError(f"Optimizer {name} not found")
    
    return registry[name](params, **kwargs)
