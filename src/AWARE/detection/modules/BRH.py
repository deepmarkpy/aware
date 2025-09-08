import torch
import torch.nn as nn

class BRH(nn.Module):
    """
    We introduce bitwise readout head(BRH), a novel module that imporves bit detection
    """
    def __init__(self, final_activation: torch.nn.modules.activation = nn.Tanh()):
        super().__init__()

        self.final_activation = final_activation

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, C, 1)
        x = self.global_avg_pool(x)

        # Comparing output pairs to make decision for every bit
        even_heads = x[:, 0::2]   # (B, C/2, 1)
        odd_heads  = x[:, 1::2]   # (B, C/2, 1)
        x = even_heads - odd_heads 

        x = self.final_activation(x)
        
        return x
