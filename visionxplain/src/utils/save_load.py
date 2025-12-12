"""
save_load.py
------------
Utilities for saving and loading models, checkpoints, and configurations.
"""

import torch
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Any


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Optional[Dict] = None,
    filepath: str = "checkpoint.pt",
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        metrics: Optional metrics dictionary
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    if metrics:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    filepath: str = "checkpoint.pt",
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state
        filepath: Path to checkpoint file
        device: Device to load on
    
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded from {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    
    return checkpoint


def save_model(model: torch.nn.Module, filepath: str, **kwargs):
    """
    Save model state dict only.
    
    Args:
        model: PyTorch model
        filepath: Path to save model
        **kwargs: Additional metadata to save
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    
    torch.save(save_dict, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(model: torch.nn.Module, filepath: str, device: Optional[torch.device] = None):
    """
    Load model state dict.
    
    Args:
        model: PyTorch model to load weights into
        filepath: Path to model file
        device: Device to load on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume it's just the state dict
        model.load_state_dict(checkpoint)
    
    print(f"✓ Model loaded from {filepath}")


def save_config(config: Dict, filepath: str):
    """
    Save configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Config saved to {filepath}")


def load_config(filepath: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        filepath: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_metrics(metrics: Dict, filepath: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        filepath: Path to save metrics
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {filepath}")


def load_metrics(filepath: str) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        filepath: Path to metrics file
    
    Returns:
        Metrics dictionary
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics

