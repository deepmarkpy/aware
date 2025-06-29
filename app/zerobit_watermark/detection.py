import torch
import torch.nn as nn

class SimpleZerobitDetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(128, 128, 5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv1d(256, 512, 3, padding=1)
        #self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(512, 1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #print(f"After conv: {x.shape}")
        
        #x = self.relu(x)
        #print(f"After relu: {x.shape}")
        
        x = self.pool(x)
        #print(f"After pool: {x.shape}")
        
        x = self.flatten(x)
        #print(f"After flatten: {x.shape}")
        
        x = self.linear(x)
        #print(f"After linear: {x.shape}")
                
        return x
