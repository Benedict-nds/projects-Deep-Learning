#!/usr/bin/env python
"""
run_benchmarks.py
-----------------
Run computational efficiency benchmarks for all models.
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.evaluation.benchmark import benchmark_inference_speed, compare_models
from src.models.baseline.cnn import SimpleCNN
from src.models.vit.vit import VisionTransformer
from src.models.hybrid.cnn_vit import HybridNet
from src.data.dataloader import create_dataloaders
from src.utils.helpers import count_parameters, get_device


def benchmark_all_models(
    model_paths: dict,
    output_file: str = "outputs/benchmarks/computational_efficiency.json",
    num_iterations: int = 100
):
    """
    Benchmark all models for computational efficiency.
    
    Args:
        model_paths: Dict mapping model_type to model_path
        output_file: Path to save benchmark results
        num_iterations: Number of iterations for speed benchmark
    """
    print("=" * 60)
    print("Computational Efficiency Benchmarking")
    print("=" * 60)
    
    device = get_device()
    print(f"\n🖥️  Device: {device}")
    
    results = {}
    
    for model_type, model_path in model_paths.items():
        if not Path(model_path).exists():
            print(f"\n⚠️  Skipping {model_type}: {model_path} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Benchmarking {model_type.upper()} Model")
        print(f"{'='*60}")
        
        try:
            # Load model
            if model_type == "cnn":
                model = SimpleCNN(num_classes=2, pretrained=False)
            elif model_type == "vit":
                model = VisionTransformer(
                    img_size=224,
                    patch_size=16,
                    num_classes=2,
                    dim=768,
                    depth=12,
                    heads=12,
                    mlp_dim=3072
                )
            elif model_type == "hybrid":
                model = HybridNet(
                    num_classes=2,
                    cnn_backbone="resnet50",
                    vit_config={
                        'img_size': 224,
                        'patch_size': 16,
                        'dim': 768,
                        'depth': 12,
                        'heads': 12,
                        'mlp_dim': 3072,
                        'dropout': 0.1
                    },
                    fusion_method="concat"
                )
            else:
                continue
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            # Count parameters
            num_params = count_parameters(model)
            model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            
            print(f"  Parameters: {num_params:,}")
            print(f"  Model size: {model_size_mb:.1f} MB")
            
            # Benchmark inference speed
            print(f"  Benchmarking inference speed ({num_iterations} iterations)...")
            speed_results = benchmark_inference_speed(
                model=model,
                input_shape=(1, 3, 224, 224),
                num_iterations=num_iterations,
                device=device
            )
            
            results[model_type] = {
                'parameters': num_params,
                'model_size_mb': round(model_size_mb, 2),
                'inference_speed': speed_results
            }
            
            print(f"  ✓ Mean inference time: {speed_results['mean_ms']:.2f} ms")
            print(f"  ✓ FPS: {speed_results['fps']:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n{'Model':<10} {'Params':<15} {'Size (MB)':<12} {'Mean (ms)':<12} {'FPS':<10}")
    print("-" * 60)
    
    for model_type, result in results.items():
        print(f"{model_type:<10} "
              f"{result['parameters']:<15,} "
              f"{result['model_size_mb']:<12.1f} "
              f"{result['inference_speed']['mean_ms']:<12.2f} "
              f"{result['inference_speed']['fps']:<10.2f}")
    
    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark model computational efficiency")
    
    parser.add_argument("--cnn_model", type=str, default="outputs/models/best_cnn_model.pt",
                        help="Path to CNN model")
    parser.add_argument("--vit_model", type=str, default="outputs/models/best_vit_model.pt",
                        help="Path to ViT model")
    parser.add_argument("--hybrid_model", type=str, default="outputs/models/best_hybrid_model.pt",
                        help="Path to Hybrid model")
    parser.add_argument("--output", type=str, default="outputs/benchmarks/computational_efficiency.json",
                        help="Output file for results")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations for speed benchmark")
    
    args = parser.parse_args()
    
    model_paths = {}
    if Path(args.cnn_model).exists():
        model_paths['cnn'] = args.cnn_model
    if Path(args.vit_model).exists():
        model_paths['vit'] = args.vit_model
    if Path(args.hybrid_model).exists():
        model_paths['hybrid'] = args.hybrid_model
    
    if not model_paths:
        print("❌ No model checkpoints found!")
        return
    
    benchmark_all_models(
        model_paths=model_paths,
        output_file=args.output,
        num_iterations=args.iterations
    )


if __name__ == "__main__":
    main()

