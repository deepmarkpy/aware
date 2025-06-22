import numpy as np
import librosa
from scipy.optimize import minimize
from scipy.signal import stft, istft
import hashlib
from typing import Optional
import torch
from torch.nn import functional as F
import time
from .detection import WatermarkDetectionNet
from ..utils.pattern_utils import bytes_to_bipolar, bytes_to_bits

class WatermarkEmbedder:
    """
    Optimization-based audio watermarking in spectral domain
    """
    
    def __init__(self, frame_length: int = 2048, hop_length: int = 512):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.embedding_bands = (1000, 4000)  # Frequency range for embedding
        self.detection_net = WatermarkDetectionNet(n_fft=frame_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_net.to(self.device)
        
    def _get_embedding_frequencies(self, sr: int, n_fft: int) -> np.ndarray:
        """Get frequency indices for watermark embedding"""
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        mask = (freqs >= self.embedding_bands[0]) & (freqs <= self.embedding_bands[1])
        return np.where(mask)[0]
    
    def _objective_function(self, watermark_coeffs: np.ndarray, 
                          original_stft: np.ndarray, 
                          watermark_pattern: np.ndarray,
                          freq_indices: np.ndarray,
                          strength: float,
                          alpha: float = 0.5) -> float:
        """
        Objective function for optimization
        
        Args:
            watermark_coeffs: Watermark coefficients to optimize
            original_stft: Original STFT magnitude
            watermark_pattern: Pseudorandom watermark pattern
            freq_indices: Frequency indices for embedding
            strength: Watermark strength
            alpha: Trade-off between imperceptibility and robustness
        """
        # Reshape watermark coefficients
        watermark_2d = watermark_coeffs.reshape(len(freq_indices), -1)
        
        # Imperceptibility term (L2 distance from original)
        imperceptibility = np.mean((watermark_2d - original_stft[freq_indices])**2)
        
        # Robustness term (correlation with pattern)
        correlation = np.corrcoef(watermark_2d.flatten(), 
                                np.tile(watermark_pattern, watermark_2d.shape[1]))[0,1]
        robustness = -(correlation * strength)**2  # Maximize correlation
        
        # Combined objective
        return alpha * imperceptibility + (1 - alpha) * robustness
    
    def _neural_objective_function(self, watermark_coeffs: np.ndarray,
                                   original_magnitude: np.ndarray,
                                   watermark_pattern: np.ndarray,
                                   freq_indices: np.ndarray) -> float:
        """
        Neural network based objective function for optimization
        
        Args:
            original_magnitude: Original STFT magnitude
            watermark_coeffs: Watermark coefficients to optimize
            watermark_pattern: Target watermark pattern
            freq_indices: Frequency indices for embedding
            
        Returns:
            Total loss value
        """
        # Reshape watermark coefficients
        watermark_2d = watermark_coeffs.reshape(len(freq_indices), -1)
        
        # Create watermarked STFT
        watermarked_magnitude = original_magnitude.copy()
        watermarked_magnitude[freq_indices] = watermark_2d
        
        # Convert to tensor and add batch dimension
        magnitude_tensor = torch.FloatTensor(watermarked_magnitude).unsqueeze(0).to(self.device)
        
        # Get neural network prediction
        with torch.no_grad():
            predicted_watermark = self.detection_net(magnitude_tensor)
        
        # Convert target pattern to tensor
        target_tensor = torch.FloatTensor(watermark_pattern).to(self.device)
        
        # Calculate loss between predicted and target watermark
        neural_loss = F.binary_cross_entropy(predicted_watermark.squeeze(), target_tensor)
        
        # Imperceptibility term (L2 distance from original)

        imperceptibility = np.mean((watermark_2d - original_magnitude[freq_indices])**2)
        
        alpha = 0.0  # Weight for imperceptibility
        beta = 1.0   # Weight for neural loss
        
        return alpha * imperceptibility + beta * neural_loss.item()
    
    def _optimization_callback(self, xk):
        """Callback function to track optimization progress"""
        import time
        
        if self._start_time is None:
            self._start_time = time.time()
        
        self._optimization_iter += 1
        
        if self._verbose: #and (self._optimization_iter % 10 == 0 or self._optimization_iter == 1):
            elapsed = time.time() - self._start_time
            
            # Calculate current loss for progress tracking
            try:
                current_loss = self._neural_objective_function(
                    xk, self._current_stft_magnitude, 
                    self._current_watermark_pattern, 
                    self._current_freq_indices
                )
                
                loss_change = ""
                if self._last_loss is not None:
                    change = current_loss - self._last_loss
                    loss_change = f" (Δ: {change:+.6f})"
                
                self._last_loss = current_loss
                
                print(f"Iter {self._optimization_iter:3d}: Loss = {current_loss:.6f}{loss_change} | "
                      f"Time: {elapsed:.1f}s | ETA: {elapsed/self._optimization_iter*(100-self._optimization_iter):.1f}s")
                
            except Exception:
                print(f"Iter {self._optimization_iter:3d}: Time: {elapsed:.1f}s")
    
    def _optimize_with_torch(self, initial_coeffs, stft_magnitude, watermark_pattern, 
                           freq_indices, bounds, max_iterations, verbose):
        """PyTorch-based optimization using Adam optimizer"""
        
        # Keep detection network in eval mode (frozen), but allow gradients to flow through
        self.detection_net.eval()
        for param in self.detection_net.parameters():
            param.requires_grad = False  # Freeze detection network weights
        
        # Convert to torch tensors
        coeffs = torch.FloatTensor(initial_coeffs).to(self.device).requires_grad_(True)
        target_pattern = torch.FloatTensor(watermark_pattern).to(self.device)
        original_stft = torch.FloatTensor(stft_magnitude).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.NAdam([coeffs], lr=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.9)
        
        # Extract bounds
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
            
            # Get neural network prediction (MUST allow gradients!)
            magnitude_tensor = watermarked_stft.unsqueeze(0)
            predicted_pattern = self.detection_net(magnitude_tensor)
            
            #input("Press Enter to continue...")
            # Calculate loss - Multiple options for tanh outputs
            predicted = predicted_pattern.squeeze()
            
            # Option 1: Hinge loss (best for binary tanh targets)
            hinge_loss = torch.mean(torch.clamp(1 - predicted * target_pattern, min=0))
            
            # Option 2: MSE loss
            mse_loss = F.mse_loss(predicted, target_pattern)
            
            # Option 3: Push-to-extremes loss (MSE + penalty for values near 0)
            push_loss = mse_loss - 0.1 * torch.mean(torch.abs(predicted))
            # Option 3b: Push-to-extremes loss for sigmoid outputs (push from 0.5)
            push_sigmoid_loss = mse_loss - 0.1 * torch.mean(torch.abs(predicted - 0.5))
            # Option 4: Sign-based loss - only cares about matching signs
            sign_loss = torch.mean(torch.clamp(-predicted * target_pattern, min=0))
            # Option 5: Cross-entropy loss
            #cross_entropy_loss = F.binary_cross_entropy(predicted, target_pattern)

            # Choose which loss to use:
            #neural_loss = sign_loss
            #neural_loss = hinge_loss 
            #neural_loss = mse_loss
            neural_loss = push_loss
            #neural_loss = cross_entropy_loss

            # Imperceptibility term
            original_coeffs = original_stft[freq_indices].flatten()
            imperceptibility = torch.mean((coeffs - original_coeffs)**2)
            
            # Combined loss
            total_loss = 1.0 * neural_loss + 0.0 * imperceptibility
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            # ENFORCE BOUNDS AFTER OPTIMIZER STEP (this is the key!)
            with torch.no_grad():
                coeffs.data = torch.clamp(coeffs.data, lower_bounds, upper_bounds)
                # Extra safety: ensure no negative magnitudes
                #coeffs.data = torch.clamp(coeffs.data, min=1e-8)
            
            # Track best
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_coeffs = coeffs.clone().detach()
            
            # Progress reporting
            if verbose and (iteration % 200 == 0 or iteration == 0):
                elapsed = time.time() - self._start_time if self._start_time else 0
                lr = optimizer.param_groups[0]['lr']
                print(f"Iter {iteration+1:3d}: Loss = {total_loss.item():.6f} | "
                      f"Neural: {neural_loss.item():.6f} | Imp: {imperceptibility.item():.6f} | "
                      f"LR: {lr:.6f} | Time: {elapsed:.1f}s")
                #print(f"Range: [{coeffs.min():.6f}, {coeffs.max():.6f}]")
                #print(f"Min position: {torch.argmin(coeffs).item()}")
                #print(f"Lower bounds: {lower_bounds.min():.6f}, {lower_bounds.max():.6f}")
                #print(f"Upper bounds: {upper_bounds.min():.6f}, {upper_bounds.max():.6f}")
                #print(f"Lower bound at min position: {lower_bounds[torch.argmin(coeffs)].item():.6f}")
                #print(f"Upper bound at min position: {upper_bounds[torch.argmin(coeffs)].item():.6f}")

                #print(f"Max difference: {np.max(watermarked_stft[freq_indices] - original_stft[freq_indices], axis=0):.6f}")
                #print("PREDICTED PATTERN: ", predicted_pattern)
                #print("TARGET PATTERN: ", target_pattern)
            
        # Create result object mimicking scipy.optimize result
        class TorchResult:
            def __init__(self, x, fun, success=True, message="Optimization completed", nfev=max_iterations):
                self.x = x
                self.fun = fun
                self.success = success
                self.message = message
                self.nfev = nfev
        
        return TorchResult(best_coeffs.cpu().numpy(), best_loss, True, "PyTorch optimization completed")
    
    def embed(self, audio: np.ndarray, 
              sample_rate: int,
              watermark: bytes,
              optimization_steps: int = 100,
              strength: float = 25.0,
              verbose: bool = True) -> np.ndarray:
        """
        Embed watermark using optimization
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            watermark: Watermark to embed
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
        
        print("stft_magnitude.shape: ", stft_magnitude.shape)

        watermark_pattern = bytes_to_bipolar(watermark)
            
        # Get embedding frequency indices
        freq_indices = self._get_embedding_frequencies(sample_rate, self.frame_length)
        
        if len(freq_indices) == 0:
            raise ValueError("No suitable frequencies found for embedding")
        
        # Initialize watermark coefficients with original values
        initial_watermark_coeffs = stft_magnitude[freq_indices].flatten()
        # Don't normalize - keep original magnitudes for proportional bounds
        
        if verbose:
            print(f"Initial coefficients shape: {initial_watermark_coeffs.shape}")
            print(f"Coefficients range: [{initial_watermark_coeffs.min():.6f}, {initial_watermark_coeffs.max():.6f}]")
        
        # Optimization bounds (keep reasonable magnitude range)
        
        #magnitude_delta_threshold = np.array([0.1*coeff * 10**(-strength/20) 
        #          for coeff in initial_watermark_coeffs])
        
        magnitude_delta_threshold = initial_watermark_coeffs * 10**(-strength/20)

        bounds = [(max(0, coeff - delta), coeff + delta) 
                  for coeff, delta in zip(initial_watermark_coeffs, magnitude_delta_threshold)]
        if verbose:
            print(f"Delta threshold range: [{magnitude_delta_threshold.min():.6f}, {magnitude_delta_threshold.max():.6f}]")
            print(f"Min lower bound: {min(bound[0] for bound in bounds):.6f}")
            print(f"Bounds created: {len(bounds)} bounds")
            print(f"Sample bounds: {bounds[:3]} ... {bounds[-3:]}")

        # Progress tracking
        self._optimization_iter = 0
        self._start_time = None
        self._last_loss = None
        self._verbose = verbose
        
        if verbose:
            print(f"Starting optimization with {len(initial_watermark_coeffs)} variables...")
            print(f"Target: {optimization_steps} iterations")
        
        # Store variables for callback access
        self._current_stft_magnitude = stft_magnitude
        self._current_watermark_pattern = watermark_pattern
        self._current_freq_indices = freq_indices
        
        # Choose optimization method
        use_torch_optimizer = True #len(initial_watermark_coeffs) > 1000
        
        if use_torch_optimizer:
            if verbose:
                print("Using PyTorch Adam optimizer (better for large problems)")
            result = self._optimize_with_torch(initial_watermark_coeffs, stft_magnitude, 
                                             watermark_pattern, freq_indices, bounds, 
                                             optimization_steps, verbose)
        else:
            if verbose:
                print("Using L-BFGS-B optimizer")
            # Optimize watermark coefficients
            result = minimize(
                self._neural_objective_function,
                initial_watermark_coeffs,
                args=(stft_magnitude, watermark_pattern, freq_indices),
                method='L-BFGS-B',
                bounds=bounds,
                callback=self._optimization_callback,
                options={
                    'maxiter': optimization_steps,
                    'maxfun': 50000,  # Increase function evaluation limit
                    'maxls': 50       # Increase line search limit
                }
            )
        
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
        #np.save('watermarked_stft.npy', watermarked_stft)
        # Reconstruct complex STFT
        watermarked_complex = watermarked_magnitude * np.exp(1j * stft_phase)
        
        # Inverse STFT
        _, watermarked_audio = istft(watermarked_complex, 
                                   nperseg=self.frame_length,
                                   noverlap=self.frame_length - self.hop_length)
        
        # Ensure same length as input
        if len(watermarked_audio) != len(audio):
            watermarked_audio = watermarked_audio[:len(audio)]
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(watermarked_audio))
        if max_val > 1.0:
           watermarked_audio = watermarked_audio / max_val
            
        return watermarked_audio
    
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
            'audio_duration': len(audio) / sample_rate
        } 