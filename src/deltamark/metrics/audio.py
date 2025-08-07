from deltamark.interfaces.metrics import BaseMetrics
import torch
import numpy as np
from pesq import pesq
import librosa

class PESQ(BaseMetrics):
    def __call__(self, output: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor, sampling_rate: int) -> float:
        if isinstance(output, torch.Tensor):
            output = output.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()
        resampled_output = librosa.resample(output, orig_sr=sampling_rate, target_sr=16000)
        resampled_target = librosa.resample(target, orig_sr=sampling_rate, target_sr=16000)
        return pesq(16000, resampled_output, resampled_target, 'wb')
    
class SNR(BaseMetrics):
    def __call__(self, output: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:

        if isinstance(output, torch.Tensor):
            output = output.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()
        
        if np.all(output == target):
            return float('inf')
        return 10 * np.log10(np.mean(output**2) / np.mean((output - target)**2))