from aware.interfaces.audio import BaseAudioProcessor
import torch

class STFT(BaseAudioProcessor):
    """
    Short-time Fourier transform
    
    Args:
        data: torch.Tensor of shape (batch_size, channels, time)
        
    Returns:
        torch.Tensor of shape (batch_size, channels, freq, time)
    """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, window: str = "hann", win_length: int = 2048):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = self.get_window(window, win_length)
    
    def get_window(self, window: str, win_length: int) -> torch.Tensor:
        if window == "hann":
            return torch.hann_window(win_length)
        elif window == "hamming":
            return torch.hamming_window(win_length)
        else:
            raise ValueError(f"Invalid window type: {window}")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, center=True, window=self.window, return_complex=True)

class ISTFT(BaseAudioProcessor):
    """
    Inverse Short-time Fourier transform
    """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, window: str = "hann", win_length: int = 2048):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = self.get_window(window, win_length)
    
    def get_window(self, window: str, win_length: int) -> torch.Tensor:
        if window == "hann":
            return torch.hann_window(win_length)
        elif window == "hamming":
            return torch.hamming_window(win_length)
        else:
            raise ValueError(f"Invalid window type: {window}")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return torch.istft(data, n_fft=self.n_fft, hop_length=self.hop_length, center=True, window=self.window)
    
class STFTDecomposer(BaseAudioProcessor):
    """
    Decompose STFT into magnitude and phase
    """
    def __call__(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.abs(data), torch.angle(data)
    
class STFTAssembler(BaseAudioProcessor):
    """
    Assemble magnitude and phase into complex tensor
    """
    def __call__(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return magnitude * torch.exp(1j * phase)
    
class STFTNormalizer(BaseAudioProcessor):
    """
    Normalize STFT magnitude to [0, 1]
    """
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data / torch.max(torch.abs(data)+1e-8)
