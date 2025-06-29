import numpy as np
import librosa
from scipy.optimize import minimize
from scipy.signal import stft, istft
from typing import Optional
import torch
from torch.nn import functional as F
import time
from .detection import SimpleZerobitDetectionNet

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

class WatermarkEmbedder:
    """
    Optimization-based audio watermarking in spectral domain
    """
    
    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.embedding_bands = (1000, 4000)
        self.detection_net = SimpleZerobitDetectionNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_net.to(self.device)
            
    def _optimize_with_adam(self, initial_coeffs, stft_magnitude, freq_indices, bounds, max_iterations, verbose):
        """PyTorch-based optimization using Adam optimizer"""
        
        # Keep detection network in eval mode (frozen), but allow gradients to flow through
        self.detection_net.eval()
        for param in self.detection_net.parameters():
            param.requires_grad = False  # Freeze detection network weights
                
        coeffs = torch.FloatTensor(initial_coeffs).to(self.device).requires_grad_(True)    
        original_stft = torch.FloatTensor(stft_magnitude).to(self.device)
        
        # Get prediction for original STFT
        with torch.no_grad():
            original_magnitude_tensor = original_stft[freq_indices].unsqueeze(0)
            original_prediction = self.detection_net(F.normalize(original_magnitude_tensor, p=2, dim=1)).squeeze()
            print("Original prediction: ", original_prediction)
            
        # Setup optimizer
        optimizer = torch.optim.NAdam([coeffs], lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)
        
        # Imperceptibility bounds
        lower_bounds = torch.FloatTensor([b[0] for b in bounds]).to(self.device)
        upper_bounds = torch.FloatTensor([b[1] for b in bounds]).to(self.device)
        
        best_loss = float('inf')
        best_coeffs = coeffs.clone()
        
        for iteration in range(max_iterations):
                   
            self._optimization_iter = iteration + 1
            
            optimizer.zero_grad()
            
            # Create watermarked spectrogram
            watermarked_stft = original_stft.clone()
            coeffs_2d = coeffs.reshape(len(freq_indices), -1)
            watermarked_stft[freq_indices] = coeffs_2d
            
            # Get neural network prediction
            magnitude_tensor = watermarked_stft[freq_indices].unsqueeze(0)
            predicted_pattern = self.detection_net(F.normalize(magnitude_tensor, p=2, dim=1))

            predicted = predicted_pattern.squeeze()            
            zero_bit_loss = -predicted # try to maximize the difference between original and predicted
            
            detection_loss = zero_bit_loss

            # Combined loss
            total_loss = detection_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            # ENFORCE BOUNDS AFTER OPTIMIZER STEP
            with torch.no_grad():
                coeffs.data = torch.clamp(coeffs.data, lower_bounds, upper_bounds)
                
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_coeffs = coeffs.clone().detach()
            
            # Progress reporting
            if verbose and (iteration % 50 == 0 or iteration == 0):
                elapsed = time.time() - self._start_time if self._start_time else 0
                lr = optimizer.param_groups[0]['lr']
                print(f"Iter {iteration+1:3d}: Loss = {total_loss.item():.6f} | predicted: {predicted.item():.6f} | "
                      f"LR: {lr:.7f} | Time: {elapsed:.1f}s")

            
        # Create result object mimicking scipy.optimize result
        class TorchResult:
            def __init__(self, x, fun, success=True, message="Optimization completed", nfev=max_iterations):
                self.x = x
                self.fun = fun
                self.success = success
                self.message = message
                self.nfev = nfev
        
        return TorchResult(best_coeffs.cpu().numpy(), best_loss, True, "Optimization completed")
    
    def embed(self, audio: np.ndarray, 
              sample_rate: int,
              optimization_steps: int = 5000,
              strength: float = 5.0,
              verbose: bool = True) -> np.ndarray:
        """
        Embed watermark using optimization
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            optimization_steps: Number of optimization iterations            
        Returns:
            Watermarked audio
        """
        # Compute STFT
        audio = audio/np.max(np.abs(audio))
        stft_result = stft(audio, nperseg=self.frame_length, 
                          noverlap=self.frame_length - self.hop_length)
        _, _, stft_complex = stft_result
        stft_magnitude = np.abs(stft_complex)
        stft_phase = np.angle(stft_complex)
        
            
        # Get embedding frequency indices
        freq_indices = self._get_embedding_frequencies(sample_rate, self.frame_length)
        
        if len(freq_indices) == 0:
            raise ValueError("No suitable frequencies found for embedding")
        
        # Initialize watermark coefficients with original values
        initial_watermark_coeffs = stft_magnitude[freq_indices].flatten()
        
        if verbose:
            print(f"Initial coefficients shape: {initial_watermark_coeffs.shape}")
            print(f"Coefficients range: [{initial_watermark_coeffs.min():.6f}, {initial_watermark_coeffs.max():.6f}]")
        
        
        magnitude_delta_threshold = initial_watermark_coeffs * 10**(-strength/20)

        bounds = [(max(0, coeff - delta), coeff + delta) 
                  for coeff, delta in zip(initial_watermark_coeffs, magnitude_delta_threshold)]
        
        # Progress tracking
        self._optimization_iter = 0
        self._start_time = None
        self._last_loss = None
        self._verbose = verbose
        
        if verbose:
            print(f"Starting optimization with {len(initial_watermark_coeffs)} variables...")
            print(f"Target: {optimization_steps} iterations")
        
        # Choose optimization method
        if verbose:
            print("Using PyTorch Adam optimizer (better for large problems)")
        
        result = self._optimize_with_adam(initial_watermark_coeffs, stft_magnitude, freq_indices, bounds, optimization_steps, verbose)
        if verbose:
            elapsed = time.time() - self._start_time if self._start_time else 0
            print(f"Optimization completed in {elapsed:.1f}s after {self._optimization_iter} iterations")
            print(f"Final loss: {result.fun:.6f}")
            print(f"Success: {result.success}")
            print(f"Optimization message: {result.message}")
            print(f"Number of function evaluations: {result.nfev}")
            if hasattr(result, 'nit'):
                print(f"Number of iterations: {result.nit}")
        
        # Apply optimized watermark
        watermarked_magnitude = stft_magnitude.copy()
        optimized_coeffs = result.x.reshape(len(freq_indices), -1)
        watermarked_magnitude[freq_indices] = optimized_coeffs
        
        # Calculate MSE for magnitudes
        mse = np.mean((watermarked_magnitude[freq_indices] - stft_magnitude[freq_indices])**2)
        max_delta = np.max(watermarked_magnitude[freq_indices] - stft_magnitude[freq_indices])
        if verbose:
            print(f"Max delta: {max_delta:.6f}")
            print(f"Magnitude MSE: {mse:.6f}")
            print(f"Magnitude range: [{optimized_coeffs.min():.6f}, {optimized_coeffs.max():.6f}]")
        
        # Calculate SNR for magnitudes
        original_power = np.mean(stft_magnitude[freq_indices]**2)
        noise = watermarked_magnitude[freq_indices] - stft_magnitude[freq_indices]
        noise_power = np.mean(noise**2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(original_power / noise_power)
            if verbose:
                print(f"Magnitude SNR: {snr_db:.2f} dB")
        
        watermarked_complex = watermarked_magnitude * np.exp(1j * stft_phase)
        
        _, watermarked_audio = istft(watermarked_complex, 
                                   nperseg=self.frame_length,
                                   noverlap=self.frame_length - self.hop_length)
        
        if len(watermarked_audio) != len(audio):
            watermarked_audio = watermarked_audio[:len(audio)]
        
        max_val = np.max(np.abs(watermarked_audio))
        if max_val > 1.0:
           watermarked_audio = watermarked_audio / max_val
            
        return watermarked_audio
    
    def _get_embedding_frequencies(self, sr: int, n_fft: int) -> np.ndarray:
        """Get frequency indices for watermark embedding"""
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mask = (freqs >= self.embedding_bands[0]) & (freqs <= self.embedding_bands[1])
        return np.where(mask)[0]
    
    def get_embedding_info(self, audio: np.ndarray, sample_rate: int) -> dict:
        """Get information about embedding parameters for given audio"""
        stft_result = stft(audio, nperseg=self.frame_length, 
                          noverlap=self.frame_length - self.hop_length)
        freqs = stft_result[0]
        
        freq_indices = self._get_embedding_frequencies(sample_rate, self.frame_length)
        embedding_freqs = freqs[freq_indices]
        
        return {
            'sample_rate': sample_rate,
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'embedding_frequency_range': self.embedding_bands,
            'num_embedding_frequencies': len(freq_indices),
            'embedding_frequencies': embedding_freqs.tolist(),
            'freq_indices': freq_indices.tolist(),
            'audio_duration': len(audio) / sample_rate
        } 