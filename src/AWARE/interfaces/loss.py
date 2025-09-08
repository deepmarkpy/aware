from abc import ABC, abstractmethod
import torch

class Loss(ABC):
    """Base class for loss functions"""
    
    @abstractmethod
    def forward(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Compute loss between predicted and target patterns
        
        Args:
            predicted: Predicted values from model
            target_pattern: Target pattern to match
            
        Returns:
            Loss tensor
        """
        pass
    
    def __call__(self, predicted: torch.Tensor, target_pattern: torch.Tensor) -> torch.Tensor:
        """Allow calling loss as function"""
        return self.forward(predicted, target_pattern)
