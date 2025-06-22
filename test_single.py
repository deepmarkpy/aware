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
from attacks import pcm_bit_depth_conversion, mp3_compression, delete_samples, resample

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
                  original_watermark: bytes, sample_rate: int) -> dict:
    """Test watermark detection against various attacks"""
    
    results = {}
    
    def detect_watermark(test_audio, attack_name="original"):
        """Helper function to detect watermark and calculate metrics"""
        try:
            # Convert audio to STFT for detection
            _, _, stft_complex = stft(test_audio, 
                                    nperseg=embedder.frame_length,
                                    noverlap=embedder.frame_length - embedder.hop_length)
            stft_magnitude = np.abs(stft_complex)
            
            # Convert to tensor
            magnitude_tensor = torch.FloatTensor(stft_magnitude).unsqueeze(0).to(device)
            embedder.detection_net.use_dropout = False
            # Detect watermark
            with torch.no_grad():
                detected_pattern = embedder.detection_net(magnitude_tensor)
            
            # Convert original watermark to bipolar pattern for comparison
            original_pattern = bytes_to_bipolar(original_watermark)
            
            # Calculate detected values
            detected_np = detected_pattern.cpu().numpy().flatten()
            
            # Convert detected values to bipolar pattern for comparison
            detected_pattern = detect_bipolar_pattern(detected_np)
            # Calculate bit error rate
            min_length = min(len(detected_np), len(original_pattern))
            bit_errors = np.sum(detected_pattern[:min_length] != original_pattern[:min_length])
            ber = bit_errors / min_length if min_length > 0 else 1.0
            
            # Calculate SNR of attacked audio vs original
            if attack_name != "original":
                signal_power = np.mean(audio**2)
                noise = test_audio - audio[:len(test_audio)]  # Handle length differences
                noise_power = np.mean(noise**2)
                attack_snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            else:
                attack_snr = float('inf')  # No attack = infinite SNR
            
            return {
                'attack': attack_name,
                'detection_success': True,
                'bit_error_rate': ber,
                'bit_errors': int(bit_errors),
                'attack_snr_db': attack_snr
            }
            
        except Exception as e:
            return {
                'attack': attack_name,
                'detection_success': False,
                'error': str(e),
                'bit_error_rate': 1.0,
                'attack_snr_db': 0.0
            }
    
    # Test 1: Original (no attack)
    print("Testing original audio...")
    results['original'] = detect_watermark(audio, "original")
    
    # Test 2: PCM bit depth conversions
    print("Testing PCM bit depth attacks...")
    for bit_depth in [8, 16, 24]:
        try:
            attacked_audio = pcm_bit_depth_conversion(audio.copy(), sample_rate, bit_depth)
            attack_name = f"pcm_{bit_depth}bit"
            results[attack_name] = detect_watermark(attacked_audio, attack_name)
        except Exception as e:
            results[f"pcm_{bit_depth}bit"] = {
                'attack': f"pcm_{bit_depth}bit",
                'detection_success': False,
                'error': str(e),
                'bit_error_rate': 1.0
            }
    
    # Test 3: MP3 compression
    print("Testing MP3 compression attacks...")
    for quality in [0, 2, 5, 9]:  # 0=best, 9=worst
        try:
            attacked_audio = mp3_compression(audio.copy(), sample_rate, quality)
            attack_name = f"mp3_q{quality}"
            results[attack_name] = detect_watermark(attacked_audio, attack_name)
        except Exception as e:
            results[f"mp3_q{quality}"] = {
                'attack': f"mp3_q{quality}",
                'detection_success': False,
                'error': str(e),
                'bit_error_rate': 1.0
            }
    
    # Test 4: Sample deletion
    print("Testing sample deletion attacks...")
    for percentage in [0.1, 0.25, 0.5]:  # 10%, 25%, 50%
        try:
            attacked_audio = delete_samples(audio.copy(), sample_rate, percentage)
            attack_name = f"delete_{int(percentage*100)}pct"
            results[attack_name] = detect_watermark(attacked_audio, attack_name)
        except Exception as e:
            results[f"delete_{int(percentage*100)}pct"] = {
                'attack': f"delete_{int(percentage*100)}pct",
                'detection_success': False,
                'error': str(e),
                'bit_error_rate': 1.0
            }
    
    # Test 5: Resampling
    print("Testing resampling attack...")
    try:
        attacked_audio = resample(audio.copy(), sample_rate)
        attack_name = "resample_16k"
        results[attack_name] = detect_watermark(attacked_audio, attack_name)
    except Exception as e:
        results["resample_16k"] = {
            'attack': "resample_16k",
            'detection_success': False,
            'error': str(e),
            'bit_error_rate': 1.0
        }
    
    # Calculate summary statistics
    successful_detections = sum(1 for r in results.values() if r.get('detection_success', False))
    total_tests = len(results)
    average_ber = np.mean([r.get('bit_error_rate', 1.0) for r in results.values()])
    
    results['summary'] = {
        'total_attacks_tested': total_tests,
        'successful_detections': successful_detections,
        'success_rate': successful_detections / total_tests if total_tests > 0 else 0.0,
        'average_ber': average_ber,
        'robust_attacks': [name for name, r in results.items() 
                          if name != 'summary' and r.get('bit_error_rate', 1.0) < 0.1]
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test watermark embedding on a single audio file')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('--watermark', default='dep', help='Watermark text (default: test_watermark)')
    parser.add_argument('--strength', type=float, default=5, help='Watermark strength (default: 25.0)')
    parser.add_argument('--steps', type=int, default=10000, help='Optimization steps (default: 100)')
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
    
    print(embedder.detection_net.get_model_info())
    input("Press Enter to continue...")
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
    # Calculate PESQ
    try:
        from pesq import pesq
        # Downsample to 16kHz for PESQ calculation
        pesq_sr = 16000
        if sr != pesq_sr:
            # Resample using librosa
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=pesq_sr)
            watermarked_16k = librosa.resample(watermarked, orig_sr=sr, target_sr=pesq_sr)
            print(f"Resampled to {pesq_sr} Hz for PESQ calculation")
        else:
            audio_16k = audio
            watermarked_16k = watermarked
        pesq_score = pesq(pesq_sr, audio_16k, watermarked_16k, 'wb')
        print(f"PESQ: {pesq_score:.3f}")
    except ImportError:
        print("PESQ library not available - install with: pip install pesq")
    except Exception as e:
        print(f"PESQ calculation failed: {e}")
    print(f"SNR: {snr:.2f} dB")
    
    # Save watermarked audio to temporary file
    temp_watermarked_path = "temp_watermarked.wav"
    sf.write(temp_watermarked_path, watermarked, sr)
    print(f"Saved watermarked audio to temporary file: {temp_watermarked_path}")
    
    # Load watermarked audio back
    loaded_watermarked, loaded_sr = sf.read(temp_watermarked_path)
    # Clean up temporary file
    import os
    os.remove(temp_watermarked_path)
    print("Cleaned up temporary file")
    
    
    # Test detection
    print("\nTesting detection against various attacks...")
    detection_results = test_detection(watermarked, embedder, device, watermark_bytes, sr)
    
    # Display original detection results
    if detection_results['original']['detection_success']:
        print(f"\n=== ORIGINAL AUDIO DETECTION ===")
        print(f"Bit Error Rate: {detection_results['original']['bit_error_rate']:.4f}")
    else:
        print(f"\nOriginal Detection failed: {detection_results['original']['error']}")
    
    # Display attack results
    print(f"\n=== ATTACK ROBUSTNESS RESULTS ===")
    for attack_name, result in detection_results.items():
        if attack_name in ['original', 'summary']:
            continue
        
        if result['detection_success']:
            print(f"{attack_name:15s}: BER={result['bit_error_rate']:.4f} | SNR={result.get('attack_snr_db', 0):.1f}dB")
        else:
            print(f"{attack_name:15s}: FAILED - {result.get('error', 'Unknown error')}")
    
    # Display summary
    if 'summary' in detection_results:
        summary = detection_results['summary']
        print(f"\n=== SUMMARY ===")
        print(f"Total attacks tested: {summary['total_attacks_tested']}")
        print(f"Successful detections: {summary['successful_detections']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Average BER: {summary['average_ber']:.4f}")
        print(f"Robust attacks (BER < 0.1): {', '.join(summary['robust_attacks']) if summary['robust_attacks'] else 'None'}")
    
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
    
    # Quality assessment
    if pesq_score > 4:
        quality = "Excellent"
    elif pesq_score > 3:
        quality = "Good"
    elif pesq_score > 2:
        quality = "Acceptable"
    else:
        quality = "Poor"
    print(f"Audio quality: {quality}")
    
    # Detection assessment
    ber = detection_results['original']['bit_error_rate']
    if ber == 0:
        detection_quality = "Perfect"
    elif ber < 0.1:
        detection_quality = "Good"
    else:
        detection_quality = "Poor"
    print(f"Detection quality: {detection_quality}")

if __name__ == "__main__":
    main() 