import torch
import torch.nn as nn
import torch.nn.functional as F
from deltamark.interfaces.loss import Loss

class HingeLoss(Loss):
    """Hinge loss - best for binary tanh targets"""
    
    def __init__(self):
        pass
    
    def forward(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Compute hinge loss"""
        return torch.mean(torch.clamp(1 - predicted * target_pattern, min=0))


class MSELoss(Loss):
    """Mean Squared Error loss"""
    
    def __init__(self):
        pass
    
    def forward(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss"""
        return F.mse_loss(predicted, target_pattern)


class PushToExtremesLoss(Loss):
    """Push-to-extremes loss - MSE + penalty for values near 0"""
    
    def __init__(self, penalty_weight=0.1):
        """
        Args:
            penalty_weight: Weight for the penalty term
        """
        self.penalty_weight = penalty_weight
    
    def forward(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Compute push-to-extremes loss"""
        mse_loss = F.mse_loss(predicted, target_pattern)
        penalty = self.penalty_weight * torch.mean(torch.abs(predicted))
        return mse_loss - penalty


class PushToExtremesSigmoidLoss(Loss):
    """Push-to-extremes loss for sigmoid outputs - pushes away from 0.5"""
    
    def __init__(self, penalty_weight=0.1):
        """
        Args:
            penalty_weight: Weight for the penalty term
        """
        self.penalty_weight = penalty_weight
    
    def forward(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Compute push-to-extremes loss for sigmoid outputs"""
        mse_loss = F.mse_loss(predicted, target_pattern)
        penalty = self.penalty_weight * torch.mean(torch.abs(predicted - 0.5))
        return mse_loss - penalty


class SignBasedLoss(Loss):
    """Sign-based loss - only cares about matching signs"""
    
    def __init__(self):
        pass
    
    def forward(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Compute sign-based loss"""
        return torch.mean(torch.clamp(-predicted * target_pattern, min=0))


class BinaryCrossEntropyLoss(Loss):
    """Binary cross-entropy loss for sigmoid outputs"""
    
    def __init__(self):
        pass
    
    def forward(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Compute binary cross-entropy loss"""
        return F.binary_cross_entropy(predicted, target_pattern)


class BERLoss(Loss):
    """Bit Error Rate (BER) loss - for hard sign matching"""
    
    def __init__(self):
        pass
    
    def forward(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Compute BER loss"""
        return torch.mean((torch.sign(predicted) != torch.sign(target_pattern)).float())


registry = {
    "hinge": HingeLoss,
    "mse": MSELoss,
    "push_extremes": PushToExtremesLoss,
    "push_sigmoid": PushToExtremesSigmoidLoss,
    "sign": SignBasedLoss,
    "bce": BinaryCrossEntropyLoss,
    "ber": BERLoss,
}   

def get_loss_fn(loss_type: str, **kwargs) -> Loss:
    """Get loss function by name
    
    Args:
        loss_type: String name of loss type
        **kwargs: Additional arguments for loss initialization
        
    Returns:
        Loss instance
    """    
    if loss_type not in registry:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(registry.keys())}")
    
    return registry[loss_type](**kwargs)
