#!/usr/bin/env python
"""
evaluate_model.py
-----------------
Evaluate a trained model on the test set.
Generates comprehensive metrics, confusion matrix, and ROC curves.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.evaluation.evaluate import evaluate_model_comprehensive
from src.utils.save_load import load_model
from src.models.baseline.cnn import SimpleCNN
from src.models.vit.vit import VisionTransformer
from src.models.hybrid.cnn_vit import HybridNet
from src.data.dataloader import create_dataloaders


def load_trained_model(model_path: str, model_type: str, num_classes: int = 2, img_size: int = 224):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_type: 'cnn' or 'vit'
        num_classes: Number of classes
        img_size: Image size
    
    Returns:
        Loaded model
    """
    print(f"\n📦 Loading {model_type.upper()} model from: {model_path}")
    
    if model_type == "cnn":
        model = SimpleCNN(num_classes=num_classes, pretrained=False)
    elif model_type == "vit":
        model = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072
        )
    elif model_type == "hybrid":
        model = HybridNet(
            num_classes=num_classes,
            cnn_backbone="resnet50",
            vit_config={
                'img_size': img_size,
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
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✓ Loaded from checkpoint (epoch {checkpoint.get('epoch', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print(f"   ✓ Loaded model state dict")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["cnn", "vit", "hybrid"], required=True,
                        help="Type of model (cnn, vit, or hybrid)")
    parser.add_argument("--data_dir", type=str, default="data/processed/chest_xray",
                        help="Path to processed dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--save_dir", type=str, default="outputs/evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model Evaluation on Test Set")
    print("=" * 60)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n🖥️  Using device: {device}")
    
    # Load test data
    print(f"\n📁 Loading test data from: {args.data_dir}")
    try:
        dataloaders, meta = create_dataloaders(
            data_root=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=False  # Disable for evaluation
        )
        test_loader = dataloaders["test"]
        class_names = list(meta["class_to_idx"].keys())
        
        print(f"   ✓ Test set loaded")
        print(f"   Classes: {class_names}")
        print(f"   Test samples: {meta['counts']['test']}")
        
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Load model
    try:
        model = load_trained_model(
            model_path=args.model_path,
            model_type=args.model_type,
            num_classes=len(class_names),
            img_size=args.img_size
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Evaluate
    print(f"\n🔍 Running evaluation...")
    print("-" * 60)
    
    try:
        results = evaluate_model_comprehensive(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            save_dir=args.save_dir
        )
        
        print("\n" + "=" * 60)
        print("✅ Evaluation Complete!")
        print("=" * 60)
        print(f"\n📊 Results saved to: {args.save_dir}")
        print(f"   - Metrics: {args.save_dir}/metrics.json")
        print(f"   - Confusion Matrix: {args.save_dir}/confusion_matrix.png")
        print(f"   - ROC Curve: {args.save_dir}/roc_curve.png")
        print(f"   - Classification Report: {args.save_dir}/classification_report.txt")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()



