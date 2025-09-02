import numpy as np
from AWARE.interfaces.watermark import BasePatternProcessor
from typing import Any

class PatternEncoder(BasePatternProcessor):
    """Utility class for watermark pattern transformations between different representations"""
    def __init__(self, mode: str = 'bits2bipolar'):
        self.mode = mode

    def __call__(self, inputs: Any) -> np.ndarray:
        if self.mode == 'bits2bipolar':
            return self._bits_to_bipolar(inputs)
        elif self.mode == 'bytes2bipolar':
            return self._bytes_to_bipolar(inputs)
        elif self.mode == 'bytes2bits':
            return self._bytes_to_bits(inputs)
        elif self.mode == 'bits':
            return inputs 
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
    def _bytes_to_bits(self, inputs: bytes) -> np.ndarray:
        """
        Convert watermark bytes to a list of bits (0s and 1s)
        
        Args:
            inputs: Watermark as bytes
            
        Returns:
            Numpy array of bits (0s and 1s)
        """
        binary_str = ''.join(format(b, '08b') for b in inputs)
        return np.array([int(bit) for bit in binary_str], dtype=np.int32)

    def _bits_to_bipolar(self, inputs: np.ndarray) -> np.ndarray:
        """
        Convert bits (0s and 1s) to bipolar representation (-1s and 1s)
        
        Args:
            inputs: Numpy array of bits (0s and 1s)
            
        Returns:
            Numpy array of bipolar values (-1s and 1s)
        """
        return np.array([2 * bit - 1 for bit in inputs], dtype=np.int32)

    def _bytes_to_bipolar(self, inputs: bytes) -> np.ndarray:
        """
        Convert watermark bytes directly to bipolar representation (-1s and 1s)
        
        Args:
            inputs: Watermark as bytes
            
        Returns:
            Numpy array of bipolar values (-1s and 1s)
        """
        bits = self._bytes_to_bits(inputs)
        return self._bits_to_bipolar(bits)
