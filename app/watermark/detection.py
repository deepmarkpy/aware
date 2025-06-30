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

class Conv2dBlock(nn.Module):
    """
    Convolutional block that preserves time dimension and only works on frequency dimension
    Uses 2D convolution with kernel size (freq_kernel, 1) to operate only on frequency axis
    """
    def __init__(self, in_channels: int, out_channels: int, freq_kernel: int = 3, activation: str = 'relu'):
        super(Conv2dBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, channels, freq, time)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.activation(x)
        return x

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
                 initial_pool_size: int = 2,
                 activation: str = 'swish',
                 watermark_length: int = 24,
                 use_dropout: bool = True):
        super(WatermarkDetectionNet, self).__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.num_blocks = num_blocks
        self.initial_pool_size = initial_pool_size
        self.use_dropout = use_dropout
        
        # Mel filter bank as first layer
        self.mel_layer = MelFilterBankLayer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        # Initial average pooling
        self.initial_pool = nn.AvgPool1d(kernel_size=initial_pool_size, stride=initial_pool_size)
        #self.initial_pool = nn.AvgPool2d(kernel_size=(initial_pool_size, initial_pool_size), stride=(initial_pool_size, initial_pool_size))

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1)
        else:
            self.dropout = nn.Identity()

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
        #self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        # Final fully connected layer
        self.fc_out = nn.Linear(channels[-1] * watermark_length, watermark_length)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
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
        #print(f"Shape before mel layer: {stft_magnitude.shape}")
        # Apply mel filter bank: (batch, freq, time) -> (batch, n_mels, time)
        x = self.mel_layer(stft_magnitude)
        #print(f"Shape after mel layer: {x.shape}")

        #x = x.unsqueeze(1) # for conv2d
        # Transpose dimensions 2 and 3 (freq and time) # for conv2d
        #x = x.transpose(1, 2)  # (batch, n_mels, time) -> (batch, time, n_mels)
        
        x = self.initial_pool(x)
        #print(f"Shape after pooling: {x.shape}")
        # Apply dropout - force training mode for dropout if use_dropout is True
        # nonzero_before = (x != 0).sum().item()
        
        if self.use_dropout:
            # Temporarily set dropout to training mode to ensure it's active
            dropout_training = self.dropout.training
            self.dropout.train()
            x = self.dropout(x)
            # Restore original training state
            self.dropout.train(dropout_training)
        else:
            print("Dropout is disabled")
        
        # Calculate nonzero values and estimate dropout rate
        # nonzero_after = (x != 0).sum().item()

        # total_elements = x.numel()
        # dropout_rate_estimate = (nonzero_before-nonzero_after) / total_elements
        # print(f"Estimated dropout rate: {dropout_rate_estimate:.3f}")
        
        # Apply convolutional blocks
        for i, block in enumerate(self.conv_blocks):
            x = block(x)  # Each block halves the time dimension due to stride=2
            #print(f"Shape after block {i}: {x.shape}")
        
        #print(f"Shape before global pooling: {x.shape}")
        # Global average pooling: (batch, channels, time) -> (batch, channels, watermark_length)
        x = self.global_avg_pool(x)
        #print(f"Shape after global pooling: {x.shape}")
        
        # Flatten: (batch, channels, watermark_length) -> (batch, channels * watermark_length)
        x = x.flatten(1)  # Flatten from dimension 1 onwards
        #print(f"Shape after flattening: {x.shape}")
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
            'use_dropout': self.use_dropout,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': f'Mel({self.n_mels}) -> AvgPool({self.initial_pool_size}) -> {"Dropout -> " if self.use_dropout else ""}{self.num_blocks}xConvBlocks -> GlobalAvgPool -> FC -> Tanh'
        }