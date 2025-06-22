import numpy as np
import soundfile as sf
import tempfile
import subprocess
import os
import platform
import time

def pcm_bit_depth_conversion(audio, sr, pcm=16):
    """
    Simulate MP3 compression with PCM bit depth conversion
    
    Args:
        audio: Input audio (float32, range -1 to 1)
        sr: Sample rate
        pcm: PCM bit depth (8, 16, 24)
        quality: MP3 quality (0=best, 9=worst)
    """
    # Convert to specified PCM bit depth and back (simulates quantization)
    if pcm == 8:
        # 8-bit signed: -128 to 127
        audio_int = np.clip(audio * 127.0, -128, 127).astype(np.int8)
        audio = audio_int.astype(np.float32) / 127.0
    elif pcm == 16:   
        # 16-bit signed: -32768 to 32767
        audio_int = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        audio = audio_int.astype(np.float32) / 32767.0
    elif pcm == 24:
        # 24-bit signed: -8388608 to 8388607
        audio_int = np.clip(audio * 8388607.0, -8388608, 8388607).astype(np.int32)
        audio = audio_int.astype(np.float32) / 8388607.0
    else:
        raise ValueError(f"Unsupported PCM bit depth: {pcm}")
    return audio

def mp3_compression(audio, sr, quality=2):
    """
    MP3 compression
    
    Args:
        audio: Input audio (float32, range -1 to 1)
        sr: Sample rate
        quality: MP3 quality (0=best, 9=worst) - 2 is good quality
    """
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("FFmpeg not found. Please install FFmpeg or skip MP3 tests.")

    def safe_delete(filepath, max_retries=5):
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

    # Create temporary files
    temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
    temp_mp3_fd, mp3_path = tempfile.mkstemp(suffix='.mp3')
    
    try:
        # Close file descriptors immediately to avoid conflicts
        os.close(temp_wav_fd)
        os.close(temp_mp3_fd)
        
        # Apply PCM conversion and save
        audio = pcm_bit_depth_conversion(audio, sr, 16)
        sf.write(temp_wav_path, audio, sr)
        
        # Convert to MP3 using ffmpeg
        result = subprocess.run(['ffmpeg', '-i', temp_wav_path, '-q:a', str(quality), mp3_path, '-y'], 
                              capture_output=True, check=True)
        
        # Small delay to ensure FFmpeg fully releases files
        time.sleep(0.1)
        
        # Load the MP3 file
        audio_data, sample_rate = sf.read(mp3_path)
        
        # Another small delay before cleanup
        time.sleep(0.1)
        
        return audio_data
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        raise e
    finally:
        # Clean up temporary files with retry logic
        safe_delete(temp_wav_path)
        safe_delete(mp3_path)

def delete_samples(audio, sr, percentage):
    """
    Delete percentage of samples from the audio
    
    Args:
        audio: Input audio (float32, range -1 to 1)
        sr: Sample rate
        percentage: Percentage of samples to delete (0-1)
    """
    samples_to_delete = int(percentage * sr)
    start_delete = np.random.randint(0, len(audio) - samples_to_delete)
    end_delete = start_delete + samples_to_delete
    
    audio = np.concatenate([
        audio[:start_delete],
        audio[end_delete:]
    ])

    return audio

def resample(audio, sr):
    """
    Resample audio to 16kHz and back to original rate using linear interpolation
    
    Args:
        audio: Input audio (float32, range -1 to 1)
        sr: Sample rate
    """
    # Downsample to 16kHz
    downsample_factor = sr // 16000
    if downsample_factor > 1:
        # Simple decimation (take every nth sample)
        downsampled_audio = audio[::downsample_factor]
        downsampled_sr = 16000
        print(f"Downsampled to {downsampled_sr} Hz: {len(downsampled_audio)} samples")
            
        # Upsample back to original rate using linear interpolation
        upsampled_audio = np.interp(
            np.arange(len(audio)), 
            np.arange(0, len(audio), downsample_factor),
            downsampled_audio
        )
        print(f"Upsampled back to {sr} Hz: {len(upsampled_audio)} samples")
        return upsampled_audio
    else:
        print(f"Audio already at or below 16kHz ({sr} Hz), skipping resampling")
        return audio

