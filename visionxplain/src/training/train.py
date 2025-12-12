"""
train.py
--------
Unified training entrypoint for VisionXplain.
Supports both CNN and ViT models with CLI arguments.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataloader import create_dataloaders
from src.data.preprocessing import get_transforms
from src.models.baseline.cnn import SimpleCNN
from src.models.vit.vit import VisionTransformer
from src.models.hybrid.cnn_vit import HybridNet
from src.training.vit_trainer import ViTTrainer
from src.training.finetune import CNNTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VisionXplain models")
    
    parser.add_argument("--model", type=str, choices=["cnn", "vit", "hybrid"], required=True,
                        help="Which model architecture to train")
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image input size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data/raw/chest_xray",
                        help="Path to chest X-ray dataset root")
    parser.add_argument("--save_dir", type=str, default="outputs/models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    return parser.parse_args()


def load_model(model_type: str, num_classes: int, img_size: int = 224):
    """
    Load model based on type.
    
    Args:
        model_type: 'cnn' or 'vit'
        num_classes: Number of output classes
        img_size: Input image size
        
    Returns:
        Model instance
    """
    if model_type == "cnn":
        return SimpleCNN(num_classes=num_classes, pretrained=True)
    
    elif model_type == "vit":
        return VisionTransformer(
            img_size=img_size,
            patch_size=16,
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            return_attentions=False
        )
    
    elif model_type == "hybrid":
        return HybridNet(
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


def main():
    """Main training function."""
    args = parse_args()
    
    print(f"\n Starting Training: {args.model.upper()} Model")
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch}")
    print(f"  - Image size: {args.img_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Data directory: {args.data_dir}")
    print("-" * 60)
    
    # Load transforms
    transforms = get_transforms(img_size=args.img_size)
    
    # Create dataloaders
    try:
        dataloaders, meta = create_dataloaders(
            data_root=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch,
            num_workers=4,
            pin_memory=True
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure your dataset is structured as:")
        print(f"  {args.data_dir}/")
        print(f"    train/")
        print(f"      NORMAL/")
        print(f"      PNEUMONIA/")
        print(f"    val/")
        print(f"      NORMAL/")
        print(f"      PNEUMONIA/")
        print(f"    test/")
        print(f"      NORMAL/")
        print(f"      PNEUMONIA/")
        return
    
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    
    # Get class names
    class_names = list(meta["class_to_idx"].keys())
    num_classes = len(class_names)
    
    print(f"\nDataset Info:")
    print(f"  - Classes: {class_names}")
    print(f"  - Train samples: {meta['counts']['train']}")
    print(f"  - Val samples: {meta['counts']['val']}")
    print(f"  - Test samples: {meta['counts']['test']}")
    
    # Load model
    model = load_model(args.model, num_classes, args.img_size)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create trainer
    if args.model == "vit":
        trainer = ViTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir
        )
    elif args.model == "hybrid":
        # Hybrid model can use CNNTrainer (similar training loop)
        trainer = CNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir
        )
    else:  # CNN
        trainer = CNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir
        )
    
    # Train (with resume support)
    if args.model == "vit":
        history = trainer.fit(resume_from=args.resume)
    else:
        history = trainer.fit(resume_from=args.resume)
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
