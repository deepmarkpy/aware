import torch

registry = {
    "reduce_lr_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosine_annealing_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "step": torch.optim.lr_scheduler.StepLR,
    "multi_step": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cyclic": torch.optim.lr_scheduler.CyclicLR,
}

def get_scheduler(name:str, optimizer: torch.optim.Optimizer, **kwargs) -> torch.optim.lr_scheduler._LRScheduler: 
    if name not in registry:
        raise ValueError(f"Scheduler {name} not found")
    return registry[name](optimizer, **kwargs)