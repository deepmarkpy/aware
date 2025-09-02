# DeltaMark
Audio watermarking with small (Δ) adversarial perturbations.

Paper title: AWARE (Adversarial WAtermarking for Robust Embedding)

## Installation
```bash
git clone https://github.com/username/repo.git
cd ./AWARE
python -m pip install -e .
```

## Basic Usage
```python
import numpy as np
import librosa
from AWARE.utils.models import *
from AWARE.service import *
from AWARE.metrics.audio import *

# 1.load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

embedder, detector = load_model()
detector = detector.to(device)


# 2.create 20-bit watermark
watermark_bits = np.random.randint(0, 2, size=20, dtype=np.int32)


# 3.read host audio
# the audio should be sampled at 44.1kHz, you can read it using soundfile:
signal, sample_rate = soundfile.read("example.wav")


# 4.embed watermark
watermarked_signal = embed_watermark(signal, 44100, watermark_bits, embedder)
# you can save it as a new one:
# soundfile.write("output.wav", watermarked_signal, 44100)


# 5.detect watermark
detected_pattern = detect_watermark(watermarked_signal, 44100, detector)


# 6.check accuracy and perceptual quality
ber_metric = BER()
ber = ber_metric(watermark_bits, detected_pattern)
print(f"BER: {ber:.2f}")

pesq_metric = PESQ()
pesq = pesq_metric(watermarked_signal, signal, 44100)
print(f"PESQ: {pesq:.2f}")