from aware.embedding.multibit_embedder import AWAREEmbedder
from aware.utils.audio import SilenceChecker
from aware.utils.watermark import PatternEncoder
from aware.utils.logger import logger
import numpy as np

def embed_watermark(audio: np.ndarray, sample_rate: int, watermark_bits:bytes | np.ndarray, model: AWAREEmbedder)->np.ndarray:
    
    """
    Embeds a given watermark in audio data and returns the watermarked audio data.

    Args:
        audio (np.ndarray): The audio data.
        sampe_rate (int): Sampling rate of the audio.
        watermark_bits (buytes | np.ndarray): The watermark bits (0/1) to embed.
        model: The embeder model.
 
    Returns:
        watermarked_audio (np.ndarray): The watermarked audio data.
    """
    pattern_preprocess_pipeline = [PatternEncoder(mode=model.pattern_mode)]
    silence_checker_pipeline = [SilenceChecker(sample_rate=sample_rate)]

    if sample_rate != 16000:
        logger.error(f"Invalid sample rate. Expected 16000Hz, got {sample_rate}Hz.")
        raise ValueError("Invalid sample rate. Expected 16000Hz.")

    watermark = watermark_bits
    for processor in pattern_preprocess_pipeline:
        watermark = processor(watermark)

    if len(watermark) != model.detection_net.output_length:
        logger.error(f"Invalid watermark length. Expected {model.detection_net.output_length}, got {len(watermark)}.")
        raise ValueError(f"Invalid watermark length.")
    
    
    if len(audio.shape) == 2 and audio.shape[1]==2: #stereo
        left_channel = audio[:, 0]
        right_channel = audio[:, 1]

        left_mx = np.max(left_channel)
        right_mx = np.max(right_channel)

        for process in silence_checker_pipeline:
            is_silent_left = process(left_channel)
            is_silent_right = process(right_channel)

        if is_silent_left == True and is_silent_right == True:
            logger.error(f"Signal you provided doesn't contain any speach. Please provide signal that contains speach.")
            raise ValueError(f"Signal you provided doesn't contain any speach. Please provide signal that contains speach.")

        left_watermarked = model.embed(left_channel, sample_rate, watermark)
        right_watermarked = model.embed(right_channel, sample_rate, watermark)

        left_watermarked = left_mx * left_watermarked
        right_watermarked = right_mx * right_watermarked

        # Combine left and right watermarked channels
        watermarked_audio = np.column_stack((left_watermarked, right_watermarked))

    elif len(audio.shape) == 1 or audio.shape[1] == 1: # mono        
        for process in silence_checker_pipeline:
            is_silent = process(audio)

        if is_silent == True:
            logger.error(f"Signal you provided doesn't contain any speach. Please provide signal that contains speach.")
            raise ValueError(f"Signal you provided doesn't contain any speach. Please provide signal that contains speach.")

        audio_mx = np.max(audio)
        
        watermarked_audio = model.embed(audio, sample_rate, watermark)

        watermarked_audio = audio_mx * watermarked_audio

    else:
        logger.error("Invalid audio shape. Expected 1D or 2D numpy array.")
        raise ValueError("Invalid audio shape. Expected 1D or 2D numpy array.")
    

    return watermarked_audio
