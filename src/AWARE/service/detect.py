import numpy as np
from AWARE.detection.multibit_detector import MultibitSTFTMagnitudeDetector
from AWARE.utils.audio import *
from AWARE.utils.watermark import *
from AWARE.utils.logger import logger


def detect_watermark(audio: np.ndarray, sample_rate: int, detector: MultibitSTFTMagnitudeDetector):
    """
    Detects the presence of a watermark in audio data and returns the detection decision, extracted watermark, and confidence statistics.

    Args:
        audio (np.ndarray): The audio data.
        samplie_rate (int): Sampling rate of the audio.
        chunk_duration (float): Duration of chunk for detection
    Returns:
        tuple:
            - bool: True if the watermark is detected with sufficient confidence, False otherwise.
            - bytes: The extracted watermark data.
            - float: The confidence score of the watermark being present in the detected values.
            - dict: The confidence scores for each segment.
    """
    pattern_postprocess_pipeline = [PatternDecoder(encoder_mode=detector.pattern_mode, threshold=detector.threshold)]
    
    if sample_rate != 44100:
        logger.error(f"Invalid sample rate. Expected 44100, got {sample_rate}.")
        raise ValueError(f"Invalid sample rate. Expected 44100")

    if len(audio.shape) == 2 and audio.shape[1] == 2: #stereo
        left_channel = audio[:, 0]
        right_channel = audio[:, 1]
        
        detected_values_left = detector.detect(left_channel, sample_rate)
        detected_values_right = detector.detect(right_channel, sample_rate)

        detected_values = []
        for i in range(len(detected_values_left)):
            if np.abs(detected_values_left[i]) > np.abs(detected_values_right[i]):
                detected_values.append(detected_values_left[i])
            else:
                detected_values.append(detected_values_right[i])

        
        watermark_bits = np.array(detected_values)
        for processor in pattern_postprocess_pipeline:
            watermark_bits = processor(watermark_bits)

        return watermark_bits

    elif len(audio.shape) == 1: # mono
        detected_values = detector.detect(audio, sample_rate)

        watermark_bits = detected_values
        for processor in pattern_postprocess_pipeline:
            watermark_bits = processor(watermark_bits)

        return watermark_bits

    else:
        logger.error("Invalid audio shape. Expected 1D or 2D numpy array.")
        raise ValueError("Invalid audio shape. Expected 1D or 2D numpy array.")
