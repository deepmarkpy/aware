import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.signal import stft
import hashlib
from typing import Dict, Optional
import os

from ..utils.mel import MelFilterBankLayer

class Conv1dBlock(nn.Module):
    """
    Convolutional block with stride 2, instance normalization and activation
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str = 'relu'):
        super(Conv1dBlock, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.instance_norm = nn.InstanceNorm1d(out_channels)
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'swish':
            self.activation = nn.SiLU()  # SiLU is PyTorch's implementation of Swish
        else:
            self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x

class WatermarkPatternMerger(nn.Module):
    """
    Module to merge watermark magnitude with STFT magnitude and apply strength scaling
    """
    def __init__(self, strength: float = 3.0):
        super(WatermarkPatternMerger, self).__init__()
        self.strength = strength
        
    def forward(self, stft_magnitude: torch.Tensor, watermark_magnitude: torch.Tensor) -> torch.Tensor:
        """
        Merge watermark magnitude with STFT magnitude
        
        Args:
            stft_magnitude: STFT magnitude tensor [batch, freq, time]
            watermark_magnitude: Watermark magnitude tensor [batch, freq, time]    
        Returns:
            Merged STFT magnitude with embedded watermark
        """
        watermark_magnitude = -1 * torch.abs(watermark_magnitude)/torch.norm(watermark_magnitude) * torch.norm(stft_magnitude) * 10**(-self.strength/20)

        return stft_magnitude + watermark_magnitude



class WatermarkDetectionNet(nn.Module):
    """
    Neural network for watermark detection in audio spectrograms
    Uses mel filter bank followed by conv blocks with global average pooling
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 num_blocks: int = 3,
                 initial_pool_size: int = 4,
                 activation: str = 'swish',
                 watermark_length: int = 24,
                 strength: float = 3.0):
        super(WatermarkDetectionNet, self).__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.num_blocks = num_blocks
        self.initial_pool_size = initial_pool_size
        self.strength = strength
        # Mel filter bank as first layer
        self.mel_layer = MelFilterBankLayer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        # Initial average pooling
        self.initial_pool = nn.AvgPool1d(kernel_size=initial_pool_size, stride=initial_pool_size)
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        # Calculate channel dimensions for each block
        channels = [n_mels] + [128 * (2**i) for i in range(num_blocks)]
        
        for i in range(num_blocks):
            block = Conv1dBlock(
                in_channels=channels[i],
                out_channels=channels[i+1],
                activation=activation
            )
            self.conv_blocks.append(block)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(watermark_length)
        
        # Final fully connected layer
        self.fc_out = nn.Linear(channels[-1] * watermark_length, watermark_length)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm1d):
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
        # Apply mel filter bank: (batch, freq, time) -> (batch, n_mels, time)
        x = self.mel_layer(stft_magnitude)
        
        # Initial average pooling: (batch, n_mels, time) -> (batch, n_mels, time//pool_size)
        x = self.initial_pool(x)
        
        # Apply convolutional blocks
        for block in self.conv_blocks:
            x = block(x)  # Each block halves the time dimension due to stride=2
        
        # print(f"Shape before global pooling: {x.shape}")
        # Global average pooling: (batch, channels, time) -> (batch, channels, watermark_length)
        x = self.global_avg_pool(x)
        # print(f"Shape after global pooling: {x.shape}")
        
        # Flatten: (batch, channels, watermark_length) -> (batch, channels * watermark_length)
        x = x.flatten(1)  # Flatten from dimension 1 onwards
        # print(f"Shape after flattening: {x.shape}")
        # Final layer: (batch, channels * watermark_length) -> (batch, watermark_length)
        x = self.fc_out(x)
        #x = self.sigmoid(x)
        x = self.tanh(x)
        return x
    
    def get_model_info(self):
        """Get information about model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
            'num_conv_blocks': self.num_blocks,
            'initial_pool_size': self.initial_pool_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': f'Mel({self.n_mels}) -> AvgPool({self.initial_pool_size}) -> {self.num_blocks}xConvBlocks -> GlobalAvgPool -> FC -> Sigmoid'
        }