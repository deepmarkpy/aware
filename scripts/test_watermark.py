import os
import yaml
import numpy as np
import librosa
from pathlib import Path

from deltamark.embedding import MultibitSTFTMagnitudeEmbedder
from deltamark.detection import MultibitSTFTMagnitudeDetector
from deltamark.utils import load_config, logger
from deltamark.metrics.audio import PESQ, SNR

import logging
logger.setLevel(logging.DEBUG)

def main():
    print("Watermark Test Pipeline")
    print("=" * 50)
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cards_dir = project_root / "src" / "deltamark" / "cards"
    audio_file = project_root / "samples" / "no-thats-not-gonna-do-it.wav"
    
    # Check if audio file exists
    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        logger.error("Please place 'no-thats-not-gonna-do-it.wav' in the samples folder")
        return
    
    # Load configurations
    logger.info("Loading configurations...")
    try:
        config = load_config(cards_dir / "config.yaml")
        logger.info("Config loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configs: {e}")
        return
    
    # Create embedder instance
    logger.info("Creating embedder...")
    try:
        embedder = MultibitSTFTMagnitudeEmbedder(
            frame_length=config.get("frame_length", 2048),
            hop_length=config.get("hop_length", 512),
            window=config.get("window", "hann"),
            win_length=config.get("win_length", 2048),
            pattern_mode=config.get("pattern_mode", "bits2bipolar"),
            embedding_bands=tuple(config.get("embedding_bands", [100, 4000])),
            tolerance_db=config.get("tolerance_db", 5.0),
            num_iterations=config.get("num_iterations", 1000),
            detection_net_cfg=config.get("detection_net_cfg", {}),
            optimizer_cfg=config.get("optimizer_cfg", {"name": "nadam", "params": {"lr": 0.1}}),
            scheduler_cfg=config.get("scheduler_cfg", {"name": "reduce_lr_on_plateau", "params": {"factor": 0.9, "patience": 500}}),
            loss=config.get("loss", "push_extremes"),
            verbose=config.get("verbose", True)
        )
        logger.info("Embedder created successfully")
        logger.info(f"   - Tolerance: {embedder.tolerance_db} dB")
        logger.info(f"   - Iterations: {embedder.num_iterations}")
        logger.info(f"   - Embedding bands: {embedder.embedding_bands} Hz")
        logger.info(f"   - Loss function: {embedder.loss}")   
        logger.info(f"   - Pattern mode: {embedder.pattern_mode}")
        logger.info(f"   - Optimizer: {embedder.optimizer_name}")
        logger.info(f"   - Scheduler: {embedder.scheduler_name}")
    except Exception as e:
        logger.error(f"Error creating embedder: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Unpack detector config
    
    logger.info("Creating detector...")
    try:
        detector = MultibitSTFTMagnitudeDetector(
            model=embedder.detection_net,
            threshold=config.get("threshold", 0.5),
            frame_length=config.get("frame_length", 2048),
            hop_length=config.get("hop_length", 512),
            window=config.get("window", "hann"),
            win_length=config.get("win_length", 2048),
            pattern_mode=config.get("pattern_mode", "bipolar")
        )
        logger.info("Detector created successfully")
        logger.info(f"   - Threshold: {detector.threshold}")       
        logger.info("Model info:")
        logger.info(detector.detection_net.get_model_info())    
    except Exception as e:
        logger.error(f"Error creating detector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load audio
    print(f"Loading audio: {audio_file.name}")
    try:
        audio, sr = librosa.load(str(audio_file), sr=44100, mono=True)
        duration = len(audio) / sr
        logger.info(f"Audio loaded successfully")
        logger.info(f"   - Duration: {duration:.2f} seconds")
        logger.info(f"   - Sample rate: {sr} Hz")
        logger.info(f"   - Samples: {len(audio)}")
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return
    
    # Generate test watermark as numpy array of bits
    watermark_length = config.get("detection_net_cfg", {}).get("watermark_length", 30)
    watermark_bits = np.random.randint(0, 2, size=watermark_length, dtype=np.int32)
    
    logger.info(f"Generated random watermark bits:")
    logger.info(f"   - Length: {watermark_length} bits")
    logger.info(f"   - Pattern: {watermark_bits}")

    # Embed watermark
    logger.info("Embedding watermark...")
    try:
        watermarked_audio = embedder.embed(audio, sr, watermark_bits)
        logger.info("Watermark embedded successfully")
        
        # Calculate embedding stats
        min_length = min(len(watermarked_audio), len(audio))
        watermarked_audio_trimmed = watermarked_audio[:min_length]
        audio_trimmed = audio[:min_length]
        
        # Initialize metrics
        pesq_metric = PESQ()
        snr_metric = SNR()
        
        # Calculate metrics
        mse = np.mean((watermarked_audio_trimmed - audio_trimmed) ** 2)
        max_diff = np.max(np.abs(watermarked_audio_trimmed - audio_trimmed))
        
        # Use proper SNR metric
        snr_value = snr_metric(watermarked_audio_trimmed, audio_trimmed)
        
        # Calculate PESQ score
        try:
            pesq_score = pesq_metric(watermarked_audio_trimmed, audio_trimmed, sr)
            logger.info(f"   - PESQ score: {pesq_score:.3f}")
        except Exception as e:
            logger.warning(f"   - PESQ calculation failed: {e}")
            pesq_score = None
        
        logger.info(f"   - MSE: {mse:.8f}")
        logger.info(f"   - SNR: {snr_value:.2f} dB")
        logger.info(f"   - Max difference: {max_diff:.6f}")
        
    except Exception as e:
        print(f"Error during embedding: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Detect watermark
    print("Detecting watermark...")
    try:
        detected_pattern = detector.detect(watermarked_audio)
        logger.info("Detection completed successfully")
        logger.info(f"   - Detected pattern: {detected_pattern}")
            
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare original and detected watermark bits
    logger.info("Comparison:")
    logger.info(f"   - Original watermark: {watermark_bits}")
    if isinstance(detected_pattern, np.ndarray):
        # Convert detected values to binary (0/1)
        detected_bits = (detected_pattern > 0).astype(int)
        logger.info(f"   - Detected pattern:   {detected_bits}")
        
        # Calculate bit accuracy
        if len(detected_bits) == len(watermark_bits):
            correct_bits = np.sum(watermark_bits == detected_bits)
            total_bits = len(watermark_bits)
            bit_accuracy = correct_bits / total_bits * 100
            logger.info(f"   - Bit accuracy: {bit_accuracy:.1f}% ({correct_bits}/{total_bits})")
        else:
            logger.warning(f"   - Length mismatch: original={len(watermark_bits)}, detected={len(detected_bits)}")
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()