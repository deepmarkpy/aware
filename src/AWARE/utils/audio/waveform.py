from AWARE.interfaces.audio import BaseAudioProcessor
import torch
import numpy as np
import webrtcvad
import resampy
from typing import Any, List, Union

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
    

class SilenceChecker(BaseAudioProcessor):

    def __init__(self, sample_rate = 16000, aggr = 3, frame_ms = 30.0, min_speech_seconds = 0.01):
        self.sample_rate = sample_rate
        self.aggr = aggr
        self.frame_ms = frame_ms
        self.min_speech_seconds = min_speech_seconds


    def __call__(self, data: np.ndarray) -> bool:
        audio = data
        
        pcm = (audio * 32767).astype(np.int16).tobytes()
        
        vad = webrtcvad.Vad(self.aggr)
        
        bytes_per_frame = int(self.sample_rate * self.frame_ms / 1000) * 2
        frames = [pcm[i:i+bytes_per_frame] for i in range(0, len(pcm), bytes_per_frame)
                if len(pcm[i:i+bytes_per_frame]) == bytes_per_frame]
        
        voiced_flags = [vad.is_speech(f, self.sample_rate) for f in frames]
        voiced_count = sum(voiced_flags)
        speech_seconds = voiced_count * (self.frame_ms / 1000.0)
        
        return speech_seconds < self.min_speech_seconds
