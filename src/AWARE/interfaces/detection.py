from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import numpy as np

class BaseDetectorNet(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class BaseDetector(ABC):    
    @abstractmethod
    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        pass
