import numpy as np
import soundfile as sf
import tempfile
import subprocess
import os
import time
import random
import pyrubberband as pyrb
from scipy.signal import butter, filtfilt
from abc import ABC, abstractmethod


class Attack(ABC):
    """Base class for audio attacks"""
    
    @abstractmethod
    def apply(self, audio, sr):
        """Apply the attack to audio
        
        Args:
            audio: Input audio (float32, range -1 to 1)
            sr: Sample rate
            
        Returns:
            Modified audio
        """
        pass


class PCMBitDepthConversion(Attack):
    """PCM bit depth conversion attack"""
    
    def __init__(self, pcm=16):
        """
        Args:
            pcm: PCM bit depth (8, 16, 24)
        """
        self.pcm = pcm
        self.name = f"pcm_{pcm}"
    
    def apply(self, audio, sr):
        """Apply PCM bit depth conversion
        
        Args:
            audio: Input audio (float32, range -1 to 1)
            sr: Sample rate
        """
        if self.pcm == 8:
            # 8-bit signed: -128 to 127
            audio_int = np.clip(audio * 127.0, -128, 127).astype(np.int8)
            audio = audio_int.astype(np.float32) / 127.0
        elif self.pcm == 16:   
            # 16-bit signed: -32768 to 32767
            audio_int = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
            audio = audio_int.astype(np.float32) / 32767.0
        elif self.pcm == 24:
            # 24-bit signed: -8388608 to 8388607
            audio_int = np.clip(audio * 8388607.0, -8388608, 8388607).astype(np.int32)
            audio = audio_int.astype(np.float32) / 8388607.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {self.pcm}")
        return audio


class MP3Compression(Attack):
    """MP3 compression attack"""
    
    def __init__(self, quality=2, pcm_bits=16):
        """
        Args:
            quality: MP3 quality (0=best, 9=worst) - 2 is good quality
            pcm_bits: PCM bit depth for pre-compression
        """
        self.quality = quality
        self.pcm_bits = pcm_bits
        self.name = f"mp3_{quality}"
        
        # Check if ffmpeg is available during initialization
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg not found. Please install FFmpeg or skip MP3 tests.")
    
    def _safe_delete(self, filepath, max_retries=5):
        """Safely delete a file with retries for Windows file locking issues"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    print(f"Warning: Could not delete {filepath} after {max_retries} attempts")
    
    def apply(self, audio, sr):
        """Apply MP3 compression
        
        Args:
            audio: Input audio (float32, range -1 to 1)
            sr: Sample rate
        """
        # Create temporary files
        temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
        temp_mp3_fd, mp3_path = tempfile.mkstemp(suffix='.mp3')
        
        try:
            # Close file descriptors immediately to avoid conflicts
            os.close(temp_wav_fd)
            os.close(temp_mp3_fd)
            
            # Apply PCM conversion and save
            pcm_converter = PCMBitDepthConversion(self.pcm_bits)
            audio = pcm_converter.apply(audio, sr)
            sf.write(temp_wav_path, audio, sr)
            
            # Convert to MP3 using ffmpeg
            subprocess.run(['ffmpeg', '-i', temp_wav_path, '-q:a', str(self.quality), mp3_path, '-y'], 
                          capture_output=True, check=True)
            
            # Small delay to ensure FFmpeg fully releases files
            time.sleep(0.1)
            
            # Load the MP3 file
            audio_data, _ = sf.read(mp3_path)
            
            # Another small delay before cleanup
            time.sleep(0.1)
            
            return audio_data
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            raise e
        finally:
            # Clean up temporary files with retry logic
            self._safe_delete(temp_wav_path)
            self._safe_delete(mp3_path)


class DeleteSamples(Attack):
    """Delete samples attack"""
    
    def __init__(self, percentage):
        """
        Args:
            percentage: Percentage of samples to delete (0-1)
        """
        self.percentage = percentage
        self.name = f"delete_{percentage}"
    
    def apply(self, audio, sr):
        """Delete percentage of samples from the audio
        
        Args:
            audio: Input audio (float32, range -1 to 1)
            sr: Sample rate
        """
        samples_to_delete = int(self.percentage * sr)
        start_delete = np.random.randint(0, len(audio) - samples_to_delete)
        end_delete = start_delete + samples_to_delete
        
        audio = np.concatenate([
            audio[:start_delete],
            audio[end_delete:]
        ])

        return audio


class TimeStretch(Attack):
    """Time stretch attack"""
    
    def __init__(self, rate = 1.0):
        """
        Args:
            `rate > 1` → playback is faster (duration decreases).
            `rate < 1` → playback is slower (duration increases).
        """
        self.rate = rate
        self.name = f"ts_{rate}"
    
    def apply(self, audio, sr):
        """
            Args:
            audio: Input audio (float32, range -1 to 1)
            sr: Sample rate
        """
        audio_stretched = pyrb.time_stretch(audio, sr, self.rate)

        return audio_stretched



class Resample(Attack):
    """Resample attack"""
    
    def __init__(self, target_sr=16000):
        """
        Args:
            target_sr: Target sample rate for downsampling
        """
        self.target_sr = target_sr
        self.name = f"resample_{target_sr}"
    
    def apply(self, audio, sr):
        """Resample audio to target rate and back using linear interpolation
        
        Args:
            audio: Input audio (float32, range -1 to 1)
            sr: Sample rate
        """
        downsample_factor = sr // self.target_sr
        if downsample_factor > 1:
            # Simple decimation (take every nth sample)
            downsampled_audio = audio[::downsample_factor]

            # Upsample back to original rate using linear interpolation
            upsampled_audio = np.interp(
                np.arange(len(audio)), 
                np.arange(0, len(audio), downsample_factor),
                downsampled_audio
            )
            
            return upsampled_audio
        
        else:
            return audio
        


class RandomBandstop(Attack):
    """Apply a random band-stop filter to audio.

    A random band (width=`band_width`) is chosen inside [min_freq, max_freq]
    (both in Hz). The chosen band is removed using a zero-phase Butterworth
    filter (filtfilt) of order `order`.

    Example:
        a = RandomBandstop(band_width=300.0)
        out = a.apply(audio, sr)
    """

    def __init__(self, band_width=200.0, min_freq=50.0, max_freq=4000.0, order=4):
        """
        Args:
            band_width (float): width of the stop band in Hz.
            min_freq (float): minimum low edge of the search range (Hz).
            max_freq (float): maximum high edge of the search range (Hz).
            order (int): Butterworth filter order (per section).
        """
        self.band_width = float(band_width)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.order = int(order)
        self.name = f"bandstop_{int(band_width)}Hz"

    def apply(self, audio, sr):
        """Apply the random bandstop.

        Args:
            audio (np.ndarray or array-like): 1D audio signal (float, -1..1 typically).
            sr (int): sample rate in Hz.

        Returns:
            np.ndarray: filtered audio (same length as input).
        """
        # ensure numpy array (use float64 for filtfilt numeric stability)
        audio_np = np.asarray(audio)
        if audio_np.ndim != 1:
            raise ValueError("RandomBandstop currently supports 1D audio arrays only.")

        # Choose random low cutoff in [50, 4000 - band_width]
        f_low = random.uniform(self.min_freq, self.max_freq - self.band_width)
        f_high = f_low + self.band_width

        # Normalize frequencies to Nyquist (half the sampling rate)
        nyq = sr / 2.0
        low = f_low / nyq
        high = f_high / nyq

        # design bandstop Butterworth and apply zero-phase filtering
        b, a = butter(self.order, [low, high], btype='bandstop')
        filtered = filtfilt(b, a, audio_np.astype(np.float64))

        # return same dtype as input if possible
        if isinstance(audio, np.ndarray):
            return filtered.astype(audio.dtype)
        else:
            return filtered


class SampleSupression(Attack):
    """Zero out samples attack"""
    
    def __init__(self, percentage):
        """
        Args:
            percentage: Percentage of samples to zero out (0-1)
        """
        self.percentage = percentage
        self.name = f"sample_supression_{percentage}"
    
    def apply(self, audio, sr):
        """Zero out percentage of samples from the audio
        
        Args:
            audio: Input audio (float32, range -1 to 1)
            sr: Sample rate
        """
        samples_to_delete = int(self.percentage * sr)
        start_delete = np.random.randint(0, len(audio) - samples_to_delete)
        end_delete = start_delete + samples_to_delete
        
        audio[start_delete : end_delete] = 0

        return audio
