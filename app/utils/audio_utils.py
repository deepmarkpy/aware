import numpy as np
import librosa
import soundfile as sf
import io
from typing import Tuple, Union

def load_audio(audio_input: Union[str, io.BytesIO], target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Load audio from file path or BytesIO object
    
    Args:
        audio_input: File path or BytesIO object containing audio data
        target_sr: Target sample rate for resampling
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        if isinstance(audio_input, io.BytesIO):
            # Load from BytesIO
            audio_array, sr = sf.read(audio_input)
        else:
            # Load from file path
            audio_array, sr = librosa.load(audio_input, sr=None)
        
        # Resample if needed
        if sr != target_sr:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = librosa.to_mono(audio_array.T)
            
        return audio_array.astype(np.float32), sr
        
    except Exception as e:
        raise ValueError(f"Error loading audio: {str(e)}")

def save_audio(audio_array: np.ndarray, sample_rate: int, output: Union[str, io.BytesIO], format: str = 'wav'):
    """
    Save audio to file or BytesIO object
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate of the audio
        output: Output file path or BytesIO object
        format: Audio format ('wav', 'flac', etc.)
    """
    try:
        # Ensure proper data type and range
        audio_array = np.clip(audio_array, -1.0, 1.0)
        
        if isinstance(output, io.BytesIO):
            sf.write(output, audio_array, sample_rate, format=format.upper())
        else:
            sf.write(output, audio_array, sample_rate)
            
    except Exception as e:
        raise ValueError(f"Error saving audio: {str(e)}")

def get_audio_features(audio_array: np.ndarray, sample_rate: int) -> dict:
    """
    Extract basic audio features for analysis
    
    Args:
        audio_array: Audio data
        sample_rate: Sample rate
        
    Returns:
        Dictionary of audio features
    """
    features = {
        'duration': len(audio_array) / sample_rate,
        'sample_rate': sample_rate,
        'channels': 1 if len(audio_array.shape) == 1 else audio_array.shape[1],
        'rms_energy': np.sqrt(np.mean(audio_array**2)),
        'max_amplitude': np.max(np.abs(audio_array)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio_array))
    }
    
    return features

def preprocess_audio(audio_array: np.ndarray, target_length: int = None) -> np.ndarray:
    """
    Preprocess audio for watermarking (normalize, pad/trim)
    
    Args:
        audio_array: Input audio
        target_length: Target length in samples (None to keep original)
        
    Returns:
        Preprocessed audio array
    """
    # Normalize to [-1, 1]
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array / np.max(np.abs(audio_array))
    
    # Pad or trim to target length
    if target_length is not None:
        if len(audio_array) < target_length:
            # Pad with zeros
            padding = target_length - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant')
        elif len(audio_array) > target_length:
            # Trim
            audio_array = audio_array[:target_length]
    
    return audio_array.astype(np.float32) 