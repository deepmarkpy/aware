import torch.nn as nn
from AWARE.interfaces.detection import BaseDetectorNet, BaseDetector
from AWARE.detection.modules import Conv1dBlock, MelFilterBankLayer, GlobalStandardize
from AWARE.utils.utils import to_tensor
from AWARE.utils.audio import *
from AWARE.utils.watermark import *

class MultibitSTFTMagnitudeDetectorNetAWARE(BaseDetectorNet):

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
                 n_filters: list[int] = [512, 1024, 1024],
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 norm_layer: str = 'instance',
                 activation: str = 'leaky_relu',
                 output_length: int = 20,
                 final_activation: str = 'tanh'):
        
        super(MultibitSTFTMagnitudeDetectorNetAWARE, self).__init__()
        
        assert len(n_filters) == num_blocks, "Number of filters must match number of blocks"
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.num_blocks = num_blocks
        self.initial_pool_size = initial_pool_size
        self.output_length = output_length
        
        self.global_norm1 = GlobalStandardize()

        # Mel filter bank as first layer
        self.mel_layer = MelFilterBankLayer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        self.instance_norm = nn.InstanceNorm1d(n_mels)
        self.global_norm2 = GlobalStandardize()

        self.initial_pool = nn.AvgPool1d(kernel_size=initial_pool_size, stride=initial_pool_stride)
        
        self.conv_blocks = nn.ModuleList()
        
        # Calculate channel dimensions for each block
        channels = [n_mels] + [n_filters[i] for i in range(num_blocks)] + [2*output_length]
        
        for i in range(num_blocks + 1):
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
        
        # Apply final activation
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
        x = stft_magnitude
        # Standardize to zero mean, unit variance
        x = self.global_norm1(x)

        #convert to mel spectrogram
        x = self.mel_layer(stft_magnitude)
        
        x = self.instance_norm(x)
        # Standardize to zero mean, unit variance
        x = self.global_norm2(x)

        # Do initial pool
        x = self.initial_pool(x)
        
        # Apply convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # (B, C, T) -> (B, C, 1)
        x = self.global_avg_pool(x)
        
        # (B, C, 1) -> (B, C) 
        x = x.squeeze(2)

        # Comparing output pairs to make decision for every bit
        even = x[:, 0::2]   # (B, C/2)
        odd  = x[:, 1::2]   # (B, C/2)
        x = even - odd 

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
