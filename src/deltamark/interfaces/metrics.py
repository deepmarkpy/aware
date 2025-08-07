from abc import ABC, abstractmethod
import torch

class BaseMetrics(ABC):
    @abstractmethod
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass