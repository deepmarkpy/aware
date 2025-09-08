from abc import ABC, abstractmethod
import numpy as np
import torch
    
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, audio: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        pass
