import torch.nn as nn

class Conv1dBlock(nn.Module):
    """
    Convolutional block with stride 2, instance normalization and activation
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, activation: str = 'relu', norm_layer: str = 'instance'):
        super(Conv1dBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm_layer = self._get_norm_layer(norm_layer)
        self.activation = self._get_activation(activation)
    
    def _get_norm_layer(self, norm_layer: str):
        if norm_layer.lower() == 'instance':
            return nn.InstanceNorm1d(self.out_channels)
        elif norm_layer.lower() == 'batch':
            return nn.BatchNorm1d(self.out_channels)
        elif norm_layer.lower() == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Invalid norm layer: {norm_layer}")
        
    def _get_activation(self, activation: str):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'swish':
            return nn.SiLU()  # SiLU is PyTorch's implementation of Swish
        else:
            return nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x
