from AWARE.interfaces.embedding import BaseEmbedder
import torch
import torch.nn as nn
import librosa
import numpy as np
import time

from AWARE.detection import MultibitSTFTMagnitudeDetectorNetAWARE
from AWARE.embedding.optimizers import get_optimizer
from AWARE.embedding.schedulers import get_scheduler
from AWARE.embedding.losses import get_loss_fn
from AWARE.utils.audio import *
from AWARE.utils.watermark import * 
from AWARE.utils.logger import logger
from AWARE.utils.utils import to_tensor

class MultibitSTFTMagnitudeEmbedder(BaseEmbedder):
    def __init__(self, frame_length: int = 2048, hop_length: int = 512, window: str = "hann", win_length: int = 2048, pattern_mode: str = "bits2bipolar", embedding_bands: tuple[int, int] = (300, 4000), tolerance_db: float = 6.0, num_iterations: int = 1000, detection_net_cfg: dict = None, optimizer_cfg: dict = None, scheduler_cfg: dict = None, loss:str = "push", verbose: bool = True):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_bands = embedding_bands
        self.tolerance_db = tolerance_db
        self.num_iterations = num_iterations
        
        self.pattern_mode = pattern_mode

        self.detection_net = MultibitSTFTMagnitudeDetectorNetAWARE(**detection_net_cfg)
        self.detection_net.eval().to(self.device) # Keep detection network in eval mode (frozen), but allow gradients to flow through
        

        self.optimizer_name = optimizer_cfg['name']
        self.optimizer_params = optimizer_cfg['params']
        self.scheduler_name = scheduler_cfg['name']
        self.scheduler_params = scheduler_cfg['params']
        self.loss = get_loss_fn(loss)

        self.verbose = verbose
        
        self.audio_preprocess_pipeline = [WaveformNormalizer(), STFT(frame_length, hop_length, window, win_length), STFTDecomposer()]
        self.audio_postprocess_pipeline = [STFTAssembler(), ISTFT(frame_length, hop_length, window, win_length), WaveformNormalizer()]
        
    def _get_embedding_frequency_indices(self, sampling_rate: int, frame_length: int) -> np.ndarray:    
        """Get frequency indices for watermark embedding"""
        freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=frame_length)
        mask = (freqs >= self.embedding_bands[0]) & (freqs <= self.embedding_bands[1])
        return np.where(mask)[0], np.where(~mask)[0] 

    def _recompute_watermarked_magnitude(self, watermarked_magnitude, stft_phase):
        y = (watermarked_magnitude, stft_phase)
        for processor in self.audio_postprocess_pipeline:
            if isinstance(y, tuple) and len(y) == 2:
                # For processors that take two arguments (like STFTAssembler)
                y = processor(*y)
            else:
                # For processors that take one argument
                y = processor(y)
        
        
        watermarked_audio = y
        
        x = watermarked_audio
        for processor in self.audio_preprocess_pipeline:
            x = processor(x)
        magnitude, _ = x

        return magnitude


    def _optimize(self, initial_coeffs: torch.Tensor, stft_magnitude: torch.Tensor, watermark_pattern: torch.Tensor, 
                           freq_indices: np.ndarray, not_freq_indices: np.ndarray, bounds: tuple[float, float], 
                           stft_phase: torch.Tensor) -> torch.Tensor:
        """PyTorch-based aversarial optimization"""
        start_time = time.time()

        for param in self.detection_net.parameters():
            param.requires_grad = False  # Freeze detection network weights
        
        coeffs = to_tensor(initial_coeffs).to(self.device).requires_grad_(True)

        watermark_pattern = watermark_pattern.to(self.device)
        stft_magnitude = stft_magnitude.to(self.device)

        # Setup optimizer
        optimizer = get_optimizer(self.optimizer_name, [coeffs], **self.optimizer_params)
        scheduler = get_scheduler(self.scheduler_name, optimizer, **self.scheduler_params)

        # Extract bounds
        lower_bounds = torch.FloatTensor([b[0] for b in bounds]).to(self.device)
        upper_bounds = torch.FloatTensor([b[1] for b in bounds]).to(self.device)

        best_loss = float('inf')
        best_coeffs = coeffs.clone()
        
        for iteration in range(self.num_iterations):    
            optimizer.zero_grad()

            # Create watermarked spectrogram
            watermarked_magnitude = stft_magnitude.clone()
            coeffs_2d = coeffs.reshape(len(freq_indices), -1)
            watermarked_magnitude[freq_indices] = coeffs_2d
            # Get neural network prediction
            watermarked_magnitude = self._recompute_watermarked_magnitude(watermarked_magnitude, stft_phase)
            watermarked_magnitude[not_freq_indices] = 0.0
            magnitude_tensor = watermarked_magnitude.unsqueeze(0)
            
            predicted_pattern= self.detection_net(magnitude_tensor).squeeze()
            
            loss = self.loss(predicted_pattern, watermark_pattern)

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # ENFORCE BOUNDS AFTER OPTIMIZER STEP
            with torch.no_grad():
                coeffs.data = torch.clamp(coeffs.data, lower_bounds, upper_bounds)
        
            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_coeffs = coeffs.clone().detach()

            if self.verbose and (iteration % 200 == 0 or iteration == 0 or iteration == 399):
                imperceptibility = torch.mean((coeffs - initial_coeffs)**2)
                if self.pattern_mode == "bits2bipolar" or self.pattern_mode == "bytes2bipolar":
                    ber = torch.mean((torch.sign(predicted_pattern) != torch.sign(watermark_pattern)).float())
                elif self.pattern_mode == "bytes2bits":
                    ber = torch.mean((predicted_pattern > 0.5) != watermark_pattern.float()).float()
                elapsed = time.time() - start_time if start_time else 0
                lr = optimizer.param_groups[0]['lr']
                logger.debug(f"Iter {iteration+1:3d}: Loss = {loss.item():.6f} | "
                      f"Neural: {loss.item():.6f} | BER: {ber.item():.6f} | Imp: {imperceptibility.item():.6f} | "
                      f"LR: {lr:.6f} | Time: {elapsed:.1f}s")

        logger.info(f"Optimization completed in {time.time() - start_time:.1f}s after {iteration+1} iterations")
        logger.info(f"Final loss: {best_loss:.6f}")
        return best_coeffs.detach().cpu()
    
    
    def embed(self, audio: np.ndarray, sample_rate: int, watermark: np.ndarray) -> np.ndarray:
        # Preprocess audio
        x = to_tensor(audio).to(self.device)

        for processor in self.audio_preprocess_pipeline:
            x = processor(x)
        magnitude, phase = x
                
        watermark_pattern = to_tensor(watermark).to(self.device)

        # Get embedding frequency indices
        freq_indices, not_freq_indices = self._get_embedding_frequency_indices(sample_rate, self.frame_length)
        
        watermark_coeffs = magnitude[freq_indices].flatten()

        # Get bounds
        magnitude_delta_threshold = watermark_coeffs * 10**(-self.tolerance_db/20)

        bounds = [(max(0, coeff - delta), coeff + delta) 
                  for coeff, delta in zip(watermark_coeffs, magnitude_delta_threshold)]
        
        if self.verbose:
            logger.info(f"Starting optimization with {len(watermark_coeffs)} variables...")
            logger.info(f"Target: {self.num_iterations} iterations")
            logger.debug(f"MAX bound: {max(max(b) for b in bounds)}")
            logger.debug(f"MIN bound: {min(min(b) for b in bounds)}")
        try:
            watermarked_coeffs = self._optimize(watermark_coeffs, magnitude, watermark_pattern, freq_indices, not_freq_indices, bounds, phase)
        except Exception as e:
            logger.error(f"Error during embedding: {e}")
            raise e

        watermarked_magnitude = magnitude.clone().detach().cpu()  
        watermarked_magnitude[freq_indices] = watermarked_coeffs.reshape(len(freq_indices), -1)

        if self.verbose:
            mse = torch.mean((watermarked_magnitude[freq_indices] - magnitude[freq_indices])**2)
            max_delta = torch.max(watermarked_magnitude[freq_indices] - magnitude[freq_indices])
            logger.debug(f"Max delta: {max_delta.item():.6f}")
            logger.debug(f"Magnitude MSE: {mse.item():.6f}")
            logger.debug(f"Magnitude range: [{watermarked_coeffs.min().item():.6f}, {watermarked_coeffs.max().item():.6f}]")

        
        # Postprocess watermarked magnitude
        y = (watermarked_magnitude, phase)
        for processor in self.audio_postprocess_pipeline:
            if isinstance(y, tuple) and len(y) == 2:
                # For processors that take two arguments (like STFTAssembler)
                y = processor(*y)
            else:
                # For processors that take one argument
                y = processor(y)
                
        watermarked_audio = y
        
        
        return watermarked_audio.detach().cpu().numpy()
    
