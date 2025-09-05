import yaml
import torch
import numpy as np

def load_config(config_path: str) -> dict:
    """
    Load a config file from a given path
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config from {config_path}: {e}")
    
def to_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Convert a numpy array or torch tensor to a torch tensor
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        return data