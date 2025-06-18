import numpy as np

def bytes_to_bits(watermark_bytes: bytes) -> np.ndarray:
    """
    Convert watermark bytes to a list of bits (0s and 1s)
    
    Args:
        watermark_bytes: Watermark as bytes
        
    Returns:
        Numpy array of bits (0s and 1s)
    """
    binary_str = ''.join(format(b, '08b') for b in watermark_bytes)
    return np.array([int(bit) for bit in binary_str], dtype=np.float32)

def bits_to_bipolar(bits: np.ndarray) -> np.ndarray:
    """
    Convert bits (0s and 1s) to bipolar representation (-1s and 1s)
    
    Args:
        bits: Numpy array of bits (0s and 1s)
        
    Returns:
        Numpy array of bipolar values (-1s and 1s)
    """
    return np.array([2 * bit - 1 for bit in bits], dtype=np.float32)

def bytes_to_bipolar(watermark_bytes: bytes) -> np.ndarray:
    """
    Convert watermark bytes directly to bipolar representation (-1s and 1s)
    
    Args:
        watermark_bytes: Watermark as bytes
        
    Returns:
        Numpy array of bipolar values (-1s and 1s)
    """
    bits = bytes_to_bits(watermark_bytes)
    return bits_to_bipolar(bits)

def detect_binary_pattern(detected_np: np.ndarray) -> np.ndarray:
    '''
    Convert detected pattern to binary pattern
    
    Args:
        detected_np: Numpy array of detected pattern
        
    Returns:
        Numpy array of binary pattern (0s and 1s)
    '''
    return (detected_np > 0.5).astype(np.float32)

def detect_bipolar_pattern(detected_np: np.ndarray) -> np.ndarray:
    '''
    Convert detected pattern to bipolar pattern
    
    Args:
        detected_np: Numpy array of detected pattern
        
    Returns:
        Numpy array of bipolar pattern (-1s and 1s)
    '''
    return 2*(detected_np > 0).astype(np.float32) - 1