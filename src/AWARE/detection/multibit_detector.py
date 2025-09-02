import torch
import numpy as np
import librosa
from AWARE.interfaces.detection import BaseDetectorNet, BaseDetector
from AWARE.utils.utils import to_tensor
from AWARE.utils.audio import *
from AWARE.utils.watermark import *


class MultibitSTFTMagnitudeDetector(BaseDetector):
    def __init__(self, model: BaseDetectorNet, threshold: float = 0.0, frame_length: int = 2048, hop_length: int = 512, window: str = "hann", win_length: int = 2048, pattern_mode: str = "bits2bipolar", embedding_bands: tuple[int, int] = (300, 4000)):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pattern_mode = pattern_mode

        self.embedding_bands = embedding_bands

        self.win_length = frame_length
        self.frame_length = frame_length
        self.hop_length = hop_length

        self.detection_net = model
        self.detection_net.eval().to(self.device)

        self.audio_preprocess_pipeline = [WaveformNormalizer(), STFT(frame_length, hop_length, window, win_length), STFTDecomposer()]
        

    def detect(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | bytes:
        x = to_tensor(audio).to(self.device)
        for processor in self.audio_preprocess_pipeline:
            x = processor(x)
        magnitude, _ = x

        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=self.frame_length)
        mask = (~((freqs >= self.embedding_bands[0]) & (freqs <= self.embedding_bands[1])))
        ids = np.where(mask)[0]
        magnitude[ids] = 0.0

        magnitude_tensor = magnitude.unsqueeze(0)
        detected_values = self.detection_net(magnitude_tensor).squeeze().detach().cpu().numpy()
        
        return detected_values
        
