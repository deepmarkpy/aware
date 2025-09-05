import numpy as np
import warnings
import torch
import torch.nn as nn

def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels"""

    scalar_input = np.isscalar(frequencies)
    frequencies = np.atleast_1d(frequencies).astype(np.float64)

    if htk:
        mels = 2595.0 * np.log10(1.0 + frequencies / 700.0)
    else:
        # Fill in the linear part
        f_min = 0.0
        f_sp = 200.0 / 3

        # Initialize the output
        mels = np.zeros_like(frequencies)

        # Linear scale below 1000 Hz
        mels = (frequencies - f_min) / f_sp

        # Log scale above 1000 Hz
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0

        # This is a boolean mask for frequencies >= min_log_hz
        log_t = frequencies >= min_log_hz

        if np.any(log_t):
            mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep

    return mels[0] if scalar_input else mels


def mel_to_hz(mels, htk=False):
    """Convert Mels to Hz"""
    # Handle scalar inputs
    scalar_input = np.isscalar(mels)
    mels = np.atleast_1d(mels).astype(np.float64)

    if htk:
        hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    else:
        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3

        # Initialize output array
        hz = np.zeros_like(mels)

        # Linear region (below min_log_mel)
        hz = f_min + f_sp * mels

        # Exponential region (above min_log_mel)
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0

        # This is a boolean mask for mels >= min_log_mel
        log_t = mels >= min_log_mel

        if np.any(log_t):
            hz[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return hz[0] if scalar_input else hz


def fft_frequencies(sr, n_fft):
    """Return the center frequencies of the FFT bins"""
    return np.linspace(0, sr / 2, int(1 + n_fft // 2), endpoint=True)


def mel_frequencies(n_mels, fmin, fmax, htk=False):
    """Return center frequencies of mel bands"""
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels, htk=htk)


def normalize(S, norm=np.inf, axis=0):
    """Normalize an array along a given axis"""
    if norm is None:
        return S

    if isinstance(norm, (int, float, np.number)):
        if norm == np.inf:
            return S / np.max(np.abs(S), axis=axis, keepdims=True)
        elif norm == -np.inf:
            return S / np.min(np.abs(S), axis=axis, keepdims=True)
        elif norm == 0:
            return S / np.sum(S != 0, axis=axis, keepdims=True)
        else:
            return S / np.sum(np.abs(S) ** norm, axis=axis, keepdims=True) ** (1.0 / norm)
    else:
        raise ValueError(f"Unsupported norm: {norm}")


def get_mel_filter_bank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm="slaney", dtype=np.float32):
    """Create a Mel filter bank (following librosa's implementation)."""
    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if isinstance(norm, str):
        if norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]
        else:
            raise ValueError(f"Unsupported norm={norm}")
    elif norm is not None:
        weights = normalize(weights, norm=norm, axis=-1)

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels."
        )

    return weights


class MelFilterBankLayer(nn.Module):
    """
    PyTorch implementation of mel filter bank layer
    """
    
    def __init__(self, sample_rate, n_fft, n_mels=128, fmin=0.0, fmax=None):
        super(MelFilterBankLayer, self).__init__()
        
        # Generate mel filter bank
        mel_basis = get_mel_filter_bank(sample_rate, n_fft, n_mels, fmin, fmax)
        
        # Register as non-trainable parameter (buffer)
        self.register_buffer(
            'mel_filter_bank', 
            torch.from_numpy(mel_basis).float()
        )
        
        self.n_mels = n_mels
        self.n_freq = n_fft // 2 + 1
    
    def forward(self, stft_output):
        """
        Parameters:
        -----------
        stft_output: torch.Tensor
            STFT magnitude output of shape (batch_size, n_freq, n_frames)
            where n_freq = n_fft//2 + 1

        Returns:
        --------
        mel_spec: torch.Tensor
            Mel spectrogram of shape (batch_size, n_mels, n_frames)
        """
        batch_size, n_freq, n_frames = stft_output.shape
        
        # Transpose to (batch_size, n_frames, n_freq)
        stft_output = stft_output.transpose(1, 2)
        
        # Reshape to (batch_size * n_frames, n_freq)
        stft_output_reshaped = stft_output.reshape(-1, n_freq)
        
        # Apply mel filter bank: (batch_size * n_frames, n_freq) @ (n_freq, n_mels)
        # mel_filter_bank is (n_mels, n_freq), so we need to transpose it
        mel_output = torch.matmul(stft_output_reshaped, self.mel_filter_bank.T)
        
        # Reshape back to (batch_size, n_frames, n_mels)
        mel_output = mel_output.reshape(batch_size, n_frames, self.n_mels)
        
        # Transpose to (batch_size, n_mels, n_frames)
        mel_output = mel_output.transpose(1, 2)
        
        # Optional: Convert to log mel (commented out to avoid numerical instability)
        # mel_output = torch.log(mel_output + 1e-8)
        
        return mel_output

# Alternative more efficient implementation using einsum
class MelFilterBankLayerEinsum(nn.Module):
    """
    More efficient version using torch.einsum
    """
    
    def __init__(self, sample_rate, n_fft, n_mels=128, fmin=0.0, fmax=None):
        super(MelFilterBankLayerEinsum, self).__init__()
        
        mel_basis = get_mel_filter_bank(sample_rate, n_fft, n_mels, fmin, fmax)
        self.register_buffer('mel_filter_bank', torch.from_numpy(mel_basis).float())
        self.n_mels = n_mels
    
    def forward(self, stft_output):
        """
        More efficient implementation using einsum
        
        Parameters:
        -----------
        stft_output: torch.Tensor (batch_size, n_freq, n_frames)
        
        Returns:
        --------
        mel_spec: torch.Tensor (batch_size, n_mels, n_frames)
        """
        # Direct matrix multiplication using einsum
        # 'bft,mf->bmt' means: batch_freq_time, mel_freq -> batch_mel_time
        mel_output = torch.einsum('bft,mf->bmt', stft_output, self.mel_filter_bank)
        
        return mel_output

# Example usage and test function
def test_mel_layer():
    """Test the mel filter bank layer"""
    sample_rate = 44100
    n_fft = 2048
    n_mels = 128
    
    # Create test data
    import torchaudio
    waveform, sr = torchaudio.load('input.wav')
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=n_fft//4, window=torch.hann_window(n_fft), return_complex=True)
    stft_mag = torch.abs(stft)
    
    # Test both implementations
    mel_layer = MelFilterBankLayer(sample_rate, n_fft, n_mels)
    mel_layer_einsum = MelFilterBankLayerEinsum(sample_rate, n_fft, n_mels)
    
    # Forward pass
    mel_spec1 = mel_layer(stft_mag)
    mel_spec2 = mel_layer_einsum(stft_mag)
    
    print(f"Input shape: {stft_mag.shape}")
    print(f"Output shape: {mel_spec1.shape}")
    print(f"Results match: {torch.allclose(mel_spec1, mel_spec2, atol=1e-10)}")

    mel_layer_tf = TFMelFilterBankLayer(sample_rate, n_fft, n_mels)
    mel_spec_tf = mel_layer_tf(stft_mag.numpy())
    mel_spec_tf = torch.from_numpy(mel_spec_tf.numpy())
    
    print(f"TF Output shape: {mel_spec_tf.shape}")
    print(f"TF Results match: {torch.allclose(mel_spec1, mel_spec_tf, atol=1e-10)}")

    return mel_spec2

if __name__ == "__main__":
    mel_spec = test_mel_layer()

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec[0].numpy(), aspect='auto', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Bins')
    plt.tight_layout()
    plt.savefig('mel_spectrogram.pdf')