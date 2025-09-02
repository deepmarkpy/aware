from deltamark.interfaces.audio import BaseAudioProcessor
import torch
from typing import Any

class WaveformNormalizer(BaseAudioProcessor):
    """
    Normalize waveform to [-1, 1]
    
    Args:
        data: torch.Tensor of shape (batch_size, channels, time)

    Returns:
        torch.Tensor of shape (batch_size, channels, time)
    """
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data/torch.max(torch.abs(data)+1e-8)


