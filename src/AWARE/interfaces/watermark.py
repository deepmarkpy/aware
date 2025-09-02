from abc import ABC, abstractmethod
from typing import Any
import torch

class BasePatternProcessor(ABC):
    @abstractmethod
    def __call__(self, data: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass