#!/usr/bin/env python3
"""
Simple zero-bit watermark detection test.
Tests one file before and after watermarking.
"""

import numpy as np
import librosa
import soundfile as sf
import argparse
import torch
from scipy.signal import stft
from app.zerobit_watermark.embedding import WatermarkEmbedder
import torch.nn.functional as F
from attacks import pcm_bit_depth_conversion, mp3_compression, delete_samples, resample

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

def detection(audio, embedder, device, sample_rate):
    """Get detection score"""
    # Convert audio to STFT
    _, _, stft_complex = stft(audio, 
                            nperseg=embedder.frame_length,
                            noverlap=embedder.frame_length - embedder.hop_length)
    stft_magnitude = np.abs(stft_complex)
    
    # Convert to tensor and detect
    freq_indices = embedder.get_embedding_info(audio, sample_rate)['freq_indices']
    magnitude_tensor = torch.FloatTensor(stft_magnitude[freq_indices]).unsqueeze(0).to(device)
    
    with torch.no_grad():
        detected_pattern = embedder.detection_net(F.normalize(magnitude_tensor, p=2, dim=1))
    
    detected_value = detected_pattern.cpu().numpy().flatten()
    return detected_value[0]

def test_attacks(audio, embedder, device, sr):
    """Test watermark detection against various attacks"""
    
    results = {}
    
    def test_single_attack(test_audio, attack_name="original"):
        """Helper function to test detection on attacked audio"""
        try:
            detection_score = detection(test_audio, embedder, device, sr)
            
            # Calculate attack SNR if not original
            if attack_name != "original":
                signal_power = np.mean(audio**2)
                noise = test_audio - audio[:len(test_audio)]
                noise_power = np.mean(noise**2)
                attack_snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            else:
                attack_snr = float('inf')
            
            return {
                'attack': attack_name,
                'detection_success': True,
                'detection_score': detection_score,
                'attack_snr_db': attack_snr
            }
            
        except Exception as e:
            return {
                'attack': attack_name,
                'detection_success': False,
                'error': str(e),
                'detection_score': 0.0,
                'attack_snr_db': 0.0
            }
    
    # Test 1: Original (no attack)
    print("Testing original audio...")
    results['original'] = test_single_attack(audio, "original")
    
    # Test 2: PCM bit depth conversions
    print("Testing PCM bit depth attacks...")
    for bit_depth in [8, 16, 24]:
        try:
            attacked_audio = pcm_bit_depth_conversion(audio.copy(), sr, bit_depth)
            attack_name = f"pcm_{bit_depth}bit"
            results[attack_name] = test_single_attack(attacked_audio, attack_name)
        except Exception as e:
            results[f"pcm_{bit_depth}bit"] = {
                'attack': f"pcm_{bit_depth}bit",
                'detection_success': False,
                'error': str(e),
                'detection_score': 0.0,
                'attack_snr_db': 0.0
            }
    
    # Test 3: MP3 compression
    print("Testing MP3 compression attacks...")
    for quality in [0, 2, 5, 9]:
        try:
            attacked_audio = mp3_compression(audio.copy(), sr, quality)
            attack_name = f"mp3_q{quality}"
            results[attack_name] = test_single_attack(attacked_audio, attack_name)
        except Exception as e:
            results[f"mp3_q{quality}"] = {
                'attack': f"mp3_q{quality}",
                'detection_success': False,
                'error': str(e),
                'detection_score': 0.0,
                'attack_snr_db': 0.0
            }
    
    # Test 4: Sample deletion
    print("Testing sample deletion attacks...")
    for percentage in [0.1, 0.25, 0.5]:  # 10%, 25%, 50%
        try:
            attacked_audio = delete_samples(audio.copy(), sr, percentage)
            attack_name = f"delete_{int(percentage*100)}pct"
            results[attack_name] = test_single_attack(attacked_audio, attack_name)
        except Exception as e:
            results[f"delete_{int(percentage*100)}pct"] = {
                'attack': f"delete_{int(percentage*100)}pct",
                'detection_success': False,
                'error': str(e),
                'detection_score': 0.0,
                'attack_snr_db': 0.0
            }
    
    # Test 5: Resampling
    print("Testing resampling attack...")
    try:
        attacked_audio = resample(audio.copy(), sr)
        attack_name = "resample_16k"
        results[attack_name] = test_single_attack(attacked_audio, attack_name)
    except Exception as e:
        results["resample_16k"] = {
            'attack': "resample_16k",
            'detection_success': False,
            'error': str(e),
            'detection_score': 0.0,
            'attack_snr_db': 0.0
        }
    
    # Calculate summary statistics
    successful_detections = sum(1 for r in results.values() if r.get('detection_success', False))
    total_tests = len(results)
    average_score = np.mean([r.get('detection_score', 0.0) for r in results.values()])
    
    results['summary'] = {
        'total_attacks_tested': total_tests,
        'successful_detections': successful_detections,
        'success_rate': successful_detections / total_tests if total_tests > 0 else 0.0,
        'average_detection_score': average_score
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Simple zero-bit detection test')
    parser.add_argument('audio_file', help='Audio file to test')
    parser.add_argument('--strength', type=float, default=5.0, help='Watermark strength')
    parser.add_argument('--optimization_steps', type=int, default=5000, help='Optimization steps')
    parser.add_argument('--output_file', type=str, default='watermarked.wav', help='Output file')
    parser.add_argument('--test_attacks', action='store_true', help='Test robustness against various attacks')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize
    embedder = WatermarkEmbedder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load original audio
    print(f"\nLoading: {args.audio_file}")
    audio, sr = librosa.load(args.audio_file, sr=None)
    print(f"Audio: {len(audio)} samples at {sr} Hz")
    
    # Test original (should be ~0)
    print("\n=== BEFORE WATERMARKING ===")
    original_detection = detection(audio, embedder, device, sr)
    print(f"Detection score: {original_detection:.6f}")
    
    # Embed watermark
    print("\n=== EMBEDDING WATERMARK ===")    
    watermarked = embedder.embed(
        audio=audio,
        sample_rate=sr,
        optimization_steps=args.optimization_steps,
        strength=args.strength,
        verbose=True
    )
    
    # Test watermarked (should be >0)
    print("\n=== AFTER WATERMARKING ===")
    watermarked_detection = detection(watermarked, embedder, device, sr)
    print(f"Detection score: {watermarked_detection:.6f}")
    
    # Calculate PESQ
    print("\n=== QUALITY METRICS ===")
    if PESQ_AVAILABLE:
        try:
            # Downsample to 16kHz for PESQ calculation
            pesq_sr = 16000
            if sr != pesq_sr:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=pesq_sr)
                watermarked_16k = librosa.resample(watermarked, orig_sr=sr, target_sr=pesq_sr)
            else:
                audio_16k = audio
                watermarked_16k = watermarked
            
            pesq_score = pesq(pesq_sr, audio_16k, watermarked_16k, 'wb')
            print(f"PESQ: {pesq_score:.3f}")
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
    else:
        print("PESQ not available - install with: pip install pesq")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Original:    {original_detection:.6f}")
    print(f"Watermarked: {watermarked_detection:.6f}")
    print(f"Difference:  {abs(watermarked_detection - original_detection):.6f}")
        
    # Save watermarked file
    sf.write(args.output_file, watermarked, sr)
    print(f"\nSaved watermarked audio: {args.output_file}")

    # Test attacks if requested
    if args.test_attacks:
        print("\n" + "="*50)
        print("TESTING ROBUSTNESS AGAINST ATTACKS")
        print("="*50)
        
        attack_results = test_attacks(watermarked, embedder, device, sr)
        
        # Display attack results
        print(f"\n=== ATTACK ROBUSTNESS RESULTS ===")
        for attack_name, result in attack_results.items():
            if attack_name in ['original', 'summary']:
                continue
            
            if result['detection_success']:
                print(f"{attack_name:15s}: Score={result['detection_score']:.6f} | SNR={result.get('attack_snr_db', 0):.1f}dB")
            else:
                print(f"{attack_name:15s}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Display summary
        if 'summary' in attack_results:
            summary = attack_results['summary']
            print(f"\n=== ATTACK SUMMARY ===")
            print(f"Total attacks tested: {summary['total_attacks_tested']}")
            print(f"Successful detections: {summary['successful_detections']}")
            print(f"Success rate: {summary['success_rate']:.1%}")
            print(f"Average detection score: {summary['average_detection_score']:.6f}")
    
if __name__ == "__main__":
    main() 