"""
helpers.py
----------
General utility functions for VisionXplain.
"""

import torch
import numpy as np
import random
from typing import Optional
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def denormalize_image(tensor: torch.Tensor, mean: tuple = (0.485, 0.456, 0.406),
                      std: tuple = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Denormalize a normalized image tensor.
    
    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization
    
    Returns:
        Denormalized image as numpy array (H, W, C) in range [0, 255]
    """
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image if batch
    
    tensor = tensor.clone()
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and change to (H, W, C)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    
    return img


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"

