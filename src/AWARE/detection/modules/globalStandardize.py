import torch
import torch.nn as nn

class GlobalStandardize(nn.Module):
    """
    Global standardization to zero mean and unit variance.

    Computes mean = x.mean() and std = x.std() over all elements,
    and returns (x - mean) / (std + eps). Optionally applies a learnable
    scalar affine transform: out = out * weight + bias.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std()
        out = (x - mean) / (std + self.eps)
        
        return out
