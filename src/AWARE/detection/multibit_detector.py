import torch.nn as nn
import torch
import numpy as np
from deltamark.interfaces.detection import BaseDetectorNet, BaseDetector
from deltamark.detection.modules import Conv1dBlock, MelFilterBankLayer
from deltamark.utils.utils import to_tensor
from deltamark.utils.audio import *
from deltamark.utils.watermark import *

class MultibitSTFTMagnitudeDetectorNet(BaseDetectorNet):

    """
    Neural network for watermark detection in audio spectrograms
    Uses mel filter bank followed by conv blocks with global average pooling
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 initial_pool_size: int = 2,
                 initial_pool_stride: int = 2,
                 num_blocks: int = 3,
                 n_filters: list[int] = [512, 1024, 2048],
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 norm_layer: str = 'instance',
                 activation: str = 'swish',
                 output_length: int = 30,
                 final_activation: str = 'tanh'):
        
        super(MultibitSTFTMagnitudeDetectorNet, self).__init__()
        
        assert len(n_filters) == num_blocks, "Number of filters must match number of blocks"
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.num_blocks = num_blocks
        self.initial_pool_size = initial_pool_size
        self.output_length = output_length
        
        # Mel filter bank as first layer
        self.mel_layer = MelFilterBankLayer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        self.initial_pool = nn.AvgPool1d(kernel_size=initial_pool_size, stride=initial_pool_stride)
        
        self.conv_blocks = nn.ModuleList()
        
        # Calculate channel dimensions for each block
        channels = [n_mels] + [n_filters[i] for i in range(num_blocks)]
        
        for i in range(num_blocks):
            block = Conv1dBlock(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_layer=norm_layer,
                in_channels=channels[i],
                out_channels=channels[i+1],
                activation=activation
            )
            self.conv_blocks.append(block)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final fully connected layer
        self.fc_out = nn.Linear(channels[-1] * 2, self.output_length)
        
        self.final_activation = self._get_activation(final_activation)

        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'swish':
            return nn.SiLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Invalid activation: {activation}")
            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
         
    def forward(self, stft_magnitude):
        """
        Forward pass through the network
        
        Args:
            stft_magnitude: torch.Tensor of shape (batch_size, freq_bins, time_frames)
            
        Returns:
            torch.Tensor of shape (batch_size, num_classes)
        """
        x = self.mel_layer(stft_magnitude)
        
        # Standardize to zero mean, unit variance
        # TODO: Extract this to a separate layer if it's still needed
        x_mean = x.mean()
        x_std = x.std()
        x = (x - x_mean) / (x_std + 1e-8)
        
        x = self.initial_pool(x)
        
        # Apply convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        
        x_avg = self.global_avg_pool(x)
        x_max = self.global_max_pool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        
        # Flatten: (batch, channels, watermark_length) -> (batch, channels * watermark_length)
        x = x.flatten(1)  # Flatten from dimension 1 onwards
        x = self.fc_out(x)
        x = self.final_activation(x)
        return x
    
    def get_model_info(self):
        """Get information about model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
            'num_blocks': self.num_blocks,
            'output_length': self.output_length,
            'final_activation': self.final_activation,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
    

class MultibitSTFTMagnitudeDetector(BaseDetector):
    def __init__(self, model: MultibitSTFTMagnitudeDetectorNet, threshold: float = 0.5, frame_length: int = 2048, hop_length: int = 512, window: str = "hann", win_length: int = 2048, pattern_mode: str = "bits2bipolar"):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pattern_mode = pattern_mode

        self.detection_net = model
        self.detection_net.eval().to(self.device)

        self.audio_preprocess_pipeline = [WaveformNormalizer(), STFT(frame_length, hop_length, window, win_length), STFTDecomposer()]
        self.pattern_postprocess_pipeline = [PatternDecoder(encoder_mode=pattern_mode)]
        
    def detect(self, audio: np.ndarray) -> np.ndarray | bytes:
        x = to_tensor(audio).to(self.device)
        for processor in self.audio_preprocess_pipeline:
            x = processor(x)
        magnitude, _ = x

        magnitude_tensor = magnitude.unsqueeze(0)
        detected_values = self.detection_net(magnitude_tensor).squeeze().detach().cpu().numpy()
        from deltamark.utils.logger import logger
        logger.debug(f"Detected values: {detected_values}")
        for processor in self.pattern_postprocess_pipeline:
            detected_values = processor(detected_values)
        
        detected_pattern = detected_values
        return detected_pattern
        
