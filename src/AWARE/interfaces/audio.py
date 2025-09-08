from abc import ABC, abstractmethod
from typing import Tuple, Any
import torch
import numpy as np

class BaseAudioProcessor(ABC):
    @abstractmethod
    def __call__(self, data: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass
