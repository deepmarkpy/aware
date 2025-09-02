from AWARE.interfaces.metrics import BaseMetrics
import torch
import numpy as np
from pesq import pesq
from pystoi import stoi
import librosa

class BER(BaseMetrics):
    def __call__(self, output: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
        if isinstance(output, torch.Tensor):
            output = output.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()

        ber = np.mean( output != target ) * 100

        return ber
    
class PESQ(BaseMetrics):
    def __call__(self, output: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor, sampling_rate: int) -> float:
        if isinstance(output, torch.Tensor):
            output = output.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()

        if len(output.shape) == 2 and output.shape[1]==2:
            output_ = np.mean(output, axis=1)
            target_ = np.mean(target, axis=1)
        else:
            output_ = output
            target_ = target

        len_ = min(len(output_), len(target_))
        output_ = output_[:len_]
        target_ = target_[:len_]
        
        resampled_output = librosa.resample(output_, orig_sr=sampling_rate, target_sr=16000)
        resampled_target = librosa.resample(target_, orig_sr=sampling_rate, target_sr=16000)
        return pesq(16000, resampled_target, resampled_output, 'wb')


class STOI(BaseMetrics):
    def __call__(self, output: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor, sampling_rate: int) -> float:
        if isinstance(output, torch.Tensor):
            output = output.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()

        if len(output.shape) == 2 and output.shape[1]==2:
            output_ = np.mean(output, axis=1)
            target_ = np.mean(target, axis=1)
        else:
            output_ = output
            target_ = target

        len_ = min(len(output_), len(target_))
        output_ = output_[:len_]
        target_ = target_[:len_]
        
        resampled_output = librosa.resample(output_, orig_sr=sampling_rate, target_sr=16000)
        resampled_target = librosa.resample(target_, orig_sr=sampling_rate, target_sr=16000)
        score = stoi(resampled_target, resampled_output, 16000)

        return float(score)



class SNR(BaseMetrics):
    def __call__(self, output: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:

        if isinstance(output, torch.Tensor):
            output = output.numpy()
        if isinstance(target, torch.Tensor):
            target = target.numpy()
        
        if len(output.shape) == 2 and output.shape[1]==2:
            output_ = np.mean(output, axis=1)
            target_ = np.mean(target, axis=1)
        else:
            output_ = output
            target_ = target

        len_ = min(len(output_), len(target_))
        output_ = output_[:len_]
        target_ = target_[:len_]

        if np.all(output_ == target_):
            return float('inf')
        return 10 * np.log10(np.mean(output_**2) / np.mean((output_ - target_)**2))
