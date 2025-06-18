import torch
import numpy as np
import hashlib
from typing import Dict, Optional
import os
from scipy.signal import stft
from torch.nn import functional as F
from .detection import WatermarkDetectionNet

class WatermarkDetector:
    """
    Audio watermark detector using neural networks
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 frame_length: int = 2048, hop_length: int = 512):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.embedding_bands = (1000, 8000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = WatermarkDetectionNet()
        self.model.to(self.device)
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize with dummy weights for demonstration
            self._init_dummy_model()
    
    def _init_dummy_model(self):
        """Initialize model with reasonable weights for demonstration"""
        # This would normally be replaced with actual trained weights
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self._init_dummy_model()
    
    def save_model(self, model_path: str):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'frame_length': self.frame_length,
            'hop_length': self.hop_length
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def _generate_watermark_pattern(self, secret_key: str, length: int) -> np.ndarray:
        """Generate same watermark pattern as embedder"""
        hash_obj = hashlib.sha256(secret_key.encode())
        seed = int.from_bytes(hash_obj.digest()[:4], byteorder='big')
        np.random.seed(seed)
        pattern = np.random.choice([-1, 1], size=length)
        return pattern.astype(np.float32)
    
    def _extract_stft_magnitude(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract STFT magnitude for the neural network"""
        # Compute STFT
        stft_result = stft(audio, nperseg=self.frame_length, 
                          noverlap=self.frame_length - self.hop_length)
        freqs, times, stft_complex = stft_result
        stft_magnitude = np.abs(stft_complex)
        
        # Normalize
        if np.max(stft_magnitude) > 0:
            stft_magnitude = stft_magnitude / np.max(stft_magnitude)
        
        return stft_magnitude
    
    def _statistical_detection(self, audio: np.ndarray, sample_rate: int, secret_key: str) -> Dict:
        """Statistical correlation-based detection as fallback"""
        # Compute STFT
        stft_result = stft(audio, nperseg=self.frame_length, 
                          noverlap=self.frame_length - self.hop_length)
        freqs, times, stft_complex = stft_result
        stft_magnitude = np.abs(stft_complex)
        
        # Get embedding frequencies
        freq_mask = (freqs >= self.embedding_bands[0]) & (freqs <= self.embedding_bands[1])
        freq_indices = np.where(freq_mask)[0]
        
        if len(freq_indices) == 0:
            return {"detected": False, "confidence": 0.0, "key_match": False}
        
        # Generate expected pattern
        pattern = self._generate_watermark_pattern(secret_key, len(freq_indices))
        
        # Extract coefficients and compute correlation
        embedding_coeffs = stft_magnitude[freq_indices].flatten()
        if len(embedding_coeffs) == 0:
            return {"detected": False, "confidence": 0.0, "key_match": False}
        
        # Compute correlation with expected pattern
        pattern_repeated = np.tile(pattern, len(embedding_coeffs) // len(pattern) + 1)[:len(embedding_coeffs)]
        
        if len(pattern_repeated) != len(embedding_coeffs):
            pattern_repeated = pattern_repeated[:len(embedding_coeffs)]
        
        correlation = np.corrcoef(embedding_coeffs, pattern_repeated)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Detection threshold
        threshold = 0.3
        detected = abs(correlation) > threshold
        confidence = min(abs(correlation) / threshold, 1.0)
        
        return {
            "detected": detected,
            "confidence": float(confidence),
            "key_match": detected,
            "correlation": float(correlation)
        }
    
    def detect(self, audio: np.ndarray, sample_rate: int, secret_key: str = "default_key") -> Dict:
        """
        Detect watermark in audio
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate
            secret_key: Secret key used for watermarking
            
        Returns:
            Detection results dictionary
        """
        try:
            # Extract STFT magnitude for neural network
            stft_magnitude = self._extract_stft_magnitude(audio, sample_rate)
            
            # Prepare input tensor (batch_size, freq_bins, time_frames)
            input_tensor = torch.FloatTensor(stft_magnitude).unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(self.device)
            
            # Neural network prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence_nn = probabilities[0, predicted_class].item()
            
            # Statistical fallback detection
            stat_result = self._statistical_detection(audio, sample_rate, secret_key)
            
            # Combine results (neural network primary, statistics as backup)
            detected_nn = predicted_class == 1
            
            # Final decision combines both methods
            final_detected = detected_nn or stat_result["detected"]
            final_confidence = max(confidence_nn, stat_result["confidence"])
            
            return {
                "detected": final_detected,
                "confidence": float(final_confidence),
                "key_match": stat_result["key_match"],
                "neural_prediction": detected_nn,
                "neural_confidence": float(confidence_nn),
                "statistical_correlation": stat_result.get("correlation", 0.0)
            }
            
        except Exception as e:
            # Fallback to statistical method only
            print(f"Neural detection failed: {e}, using statistical method")
            return self._statistical_detection(audio, sample_rate, secret_key)
    
    def get_detection_info(self) -> Dict:
        """Get detector configuration information"""
        return {
            'model_type': 'CNN + Statistical',
            'device': str(self.device),
            'frame_length': self.frame_length,
            'hop_length': self.hop_length,
            'embedding_frequency_range': self.embedding_bands,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        } 