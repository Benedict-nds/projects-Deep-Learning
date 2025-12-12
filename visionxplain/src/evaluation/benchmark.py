"""
benchmark.py
------------
Benchmarking utilities for model performance comparison.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional
from src.training.metrics import evaluate_model


def benchmark_inference_speed(
    model: torch.nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    num_iterations: int = 100,
    device: Optional[torch.device] = None,
    warmup: int = 10
) -> Dict:
    """
    Benchmark model inference speed.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, height, width)
        num_iterations: Number of inference iterations
        device: Device to run on
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with timing statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize if CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'fps': float(1000 / np.mean(times))
    }


def compare_models(
    models: Dict[str, torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary mapping model names to models
        dataloader: DataLoader for evaluation
        device: Device to run on
        class_names: List of class names
    
    Returns:
        Dictionary with comparison results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Evaluate metrics
        eval_results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            class_names=class_names
        )
        
        # Benchmark speed
        speed_results = benchmark_inference_speed(
            model=model,
            device=device
        )
        
        results[name] = {
            'metrics': eval_results['metrics'],
            'speed': speed_results
        }
    
    return results

