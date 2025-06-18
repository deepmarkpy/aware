import numpy as np
import librosa
import soundfile as sf
import time
import argparse
from pathlib import Path
from app.watermark.embedding import WatermarkEmbedder
import torch
from scipy.signal import stft
from app.utils.pattern_utils import bytes_to_bipolar, bytes_to_bits, detect_bipolar_pattern, detect_binary_pattern

def calculate_snr(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio in dB"""
    watermarked = watermarked[:len(original)]
    signal_power = np.mean(original**2)
    noise = watermarked - original
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def test_detection(audio: np.ndarray, embedder: WatermarkEmbedder, device: torch.device,
                  original_watermark: bytes) -> dict:
    """Test watermark detection"""
    try:
        # Convert audio to STFT for detection
        _, _, stft_complex = stft(audio, 
                                nperseg=embedder.frame_length,
                                noverlap=embedder.frame_length - embedder.hop_length)
        stft_magnitude = np.abs(stft_complex)
        stft_magnitude = np.load('watermarked_stft.npy')

        # Convert to tensor
        stft_magnitude = torch.FloatTensor(stft_magnitude).unsqueeze(0).to(device)
        
        # Detect watermark
        with torch.no_grad():
            detected_pattern = embedder.detection_net(stft_magnitude)
        
        original_pattern = bytes_to_bipolar(original_watermark)

        print("DETECTED PATTERN: ", detected_pattern)
        # Calculate detection metrics
        detected_np = detected_pattern.cpu().numpy().flatten()
        
        # Threshold detected pattern to binary
        # detected_binary = 2*(detected_np > 0).astype(np.float32) - 1
        # detected_binary = (detected_np > 0.5).astype(np.float32)
        detected_pattern = detect_bipolar_pattern(detected_np)
        print(detected_pattern)
        print(original_pattern[:len(detected_pattern)])
        # Calculate bit error rate
        bit_errors = np.sum(detected_pattern != original_pattern[:len(detected_pattern)])
        ber = bit_errors / len(detected_pattern) if len(detected_pattern) > 0 else 1.0
        
        # Calculate correlation
        correlation = np.corrcoef(detected_np, original_pattern[:len(detected_np)])[0, 1] if len(detected_np) > 0 else 0.0
        
        return {
            'success': True,
            'bit_error_rate': ber,
            'correlation': correlation,
            'detected_length': len(detected_np),
            'original_length': len(original_pattern)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'bit_error_rate': 1.0,
            'correlation': 0.0
        }

def main():
    parser = argparse.ArgumentParser(description='Test watermark embedding on a single audio file')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('--watermark', default='dep', help='Watermark text (default: test_watermark)')
    parser.add_argument('--strength', type=float, default=3, help='Watermark strength (default: 25.0)')
    parser.add_argument('--steps', type=int, default=1000, help='Optimization steps (default: 100)')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--quiet', action='store_true', help='Disable optimization progress output')
    
    args = parser.parse_args()
    
    # Convert watermark to bytes
    watermark_bytes = args.watermark.encode('utf-8')
    
    print(f"Testing watermark embedding:")
    print(f"  Input file: {args.input_file}")
    print(f"  Watermark: '{args.watermark}' ({len(watermark_bytes)} bytes)")
    print(f"  Strength: {args.strength}")
    print(f"  Optimization steps: {args.steps}")
    print()
    
    # Load audio
    try:
        audio, sr = librosa.load(args.input_file, sr=None)
        print(f"Loaded audio: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f}s)")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    # Initialize embedder and detection network
    embedder = WatermarkEmbedder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Debug: Check embedding info
    embedding_info = embedder.get_embedding_info(audio, sr)
    print(f"\nEmbedding info:")
    print(f"  Frequency range: {embedding_info['embedding_frequency_range']} Hz")
    print(f"  Number of embedding frequencies: {embedding_info['num_embedding_frequencies']}")
    
    # Embed watermark
    print("\nEmbedding watermark...")
    start_time = time.time()
    
    try:
        watermarked = embedder.embed(
            audio=audio,
            sample_rate=sr,
            watermark=watermark_bytes,
            optimization_steps=args.steps,
            strength=args.strength,
            verbose=not args.quiet
        )
        embedding_time = time.time() - start_time
        print(f"Embedding completed in {embedding_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during embedding: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Calculate SNR
    snr = calculate_snr(audio, watermarked)
    print(f"SNR: {snr:.2f} dB")
    
    # Test detection
    print("\nTesting detection...")
    detection_results = test_detection(watermarked, embedder, device, watermark_bytes)
    
    if detection_results['success']:
        print(f"Bit Error Rate: {detection_results['bit_error_rate']:.4f}")
        print(f"Correlation: {detection_results['correlation']:.4f}")
        print(f"Pattern lengths - Detected: {detection_results['detected_length']}, Original: {detection_results['original_length']}")
    else:
        print(f"Detection failed: {detection_results['error']}")
    
    # Save output file
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_file)
        output_path = f"{input_path.stem}_watermarked.wav"
    
    try:
        sf.write(output_path, watermarked, sr)
        print(f"\nWatermarked audio saved to: {output_path}")
    except Exception as e:
        print(f"Error saving output: {e}")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Embedding time: {embedding_time:.2f}s")
    print(f"SNR: {snr:.2f} dB")
    print(f"BER: {detection_results.get('bit_error_rate', 'N/A'):.4f}")
    
    # Quality assessment
    if snr > 40:
        quality = "Excellent"
    elif snr > 30:
        quality = "Good"
    elif snr > 20:
        quality = "Acceptable"
    else:
        quality = "Poor"
    print(f"Audio quality: {quality}")
    
    # Detection assessment
    ber = detection_results.get('bit_error_rate', 1.0)
    if ber == 0:
        detection_quality = "Perfect"
    elif ber < 0.1:
        detection_quality = "Good"
    else:
        detection_quality = "Poor"
    print(f"Detection quality: {detection_quality}")

if __name__ == "__main__":
    main() 