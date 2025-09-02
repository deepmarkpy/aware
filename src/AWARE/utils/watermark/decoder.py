import numpy as np
from deltamark.interfaces.watermark import BasePatternProcessor

class PatternDecoder(BasePatternProcessor):
    """Utility class for decoding detected watermark patterns"""
    def __init__(self, threshold: float = 0.5, encoder_mode: str = 'bits2bipolar'):
        """
        Args:
            threshold: Decision threshold for binary conversion
            encoder_mode: Mode of the encoder to use for decoding
        """
        self.threshold = threshold
        self.encoder_mode = encoder_mode

    def __call__(self, detected_values: np.ndarray) -> np.ndarray:
        if self.encoder_mode == 'bits2bipolar':
            return self._bipolar_to_bits(self._detect_bipolar(detected_values, self.threshold))
        elif self.encoder_mode == 'bytes2bipolar':
            return self._bipolar_to_bytes(self._detect_bipolar(detected_values, self.threshold))
        elif self.encoder_mode == 'bytes2bits':
            return self._bits_to_bytes(self._detect_binary(detected_values, self.threshold))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _detect_binary(self, detected_values: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert detected values to binary pattern (0s and 1s)
        
        Args:
            detected_values: Numpy array of detected pattern values
            threshold: Decision threshold for binary conversion
            
        Returns:
            Numpy array of binary pattern (0s and 1s)
        """
        return (detected_values > threshold).astype(np.float32)

    def _detect_bipolar(self, detected_values: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """
        Convert detected values to bipolar pattern (-1s and 1s)
        
        Args:
            detected_values: Numpy array of detected pattern values
            threshold: Decision threshold for bipolar conversion
            
        Returns:
            Numpy array of bipolar pattern (-1s and 1s)
        """
        return 2*(detected_values > threshold).astype(np.float32) - 1

    def _bits_to_bytes(self, detected_values: np.ndarray) -> bytes:
        """
        Convert detected values to bytes
        """ 
        return bytes([int(bit) for bit in detected_values])
    
    def _bipolar_to_bits(self, detected_values: np.ndarray) -> np.ndarray:
        """
        Convert detected values to bits
        """
        return (detected_values > 0).astype(np.float32)
    
    def _bipolar_to_bytes(self, detected_values: np.ndarray) -> bytes:
        """
        Convert detected values to bytes
        """
        return self._bits_to_bytes(self._bipolar_to_bits(detected_values))
