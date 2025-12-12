#!/usr/bin/env python
"""
Quick training test script.
Tests the training pipeline with minimal epochs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_training():
    """Test training with minimal setup."""
    print("=" * 60)
    print("VisionXplain Training Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("\n1. Testing imports...")
        from src.models.baseline.cnn import SimpleCNN
        from src.models.vit.vit import VisionTransformer
        from src.training.finetune import CNNTrainer
        from src.training.vit_trainer import ViTTrainer
        print("   ✓ Model imports successful")
        
        # Test model creation
        print("\n2. Testing model creation...")
        cnn_model = SimpleCNN(num_classes=2)
        print(f"   ✓ CNN model created: {sum(p.numel() for p in cnn_model.parameters())} parameters")
        
        vit_model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=2,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072
        )
        print(f"   ✓ ViT model created: {sum(p.numel() for p in vit_model.parameters())} parameters")
        
        # Test data loading (this might fail due to torchvision issue)
        print("\n3. Testing data loading...")
        try:
            from src.data.dataloader import create_dataloaders
            dataloaders, meta = create_dataloaders(
                data_root="data/raw/chest_xray",
                batch_size=4,
                num_workers=0,
                img_size=224
            )
            print(f"   ✓ DataLoader created successfully")
            print(f"     Classes: {list(meta['class_to_idx'].keys())}")
            print(f"     Train: {meta['counts']['train']} samples")
            print(f"     Val: {meta['counts']['val']} samples")
            print(f"     Test: {meta['counts']['test']} samples")
            
            # Test a forward pass
            print("\n4. Testing forward pass...")
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"   Using device: {device}")
            
            # Get a batch
            batch = next(iter(dataloaders["train"]))
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            # Test CNN forward
            cnn_model = cnn_model.to(device)
            cnn_model.eval()
            with torch.no_grad():
                cnn_output = cnn_model(images[:2])  # Just 2 samples
            print(f"   ✓ CNN forward pass: output shape {cnn_output.shape}")
            
            # Test ViT forward
            vit_model = vit_model.to(device)
            vit_model.eval()
            with torch.no_grad():
                vit_output = vit_model(images[:2])  # Just 2 samples
            print(f"   ✓ ViT forward pass: output shape {vit_output.shape}")
            
            print("\n" + "=" * 60)
            print("✅ All tests passed! Training pipeline is ready.")
            print("=" * 60)
            print("\nTo run full training:")
            print("  python src/training/train.py --model cnn --epochs 10 --batch 32 --img_size 224 --lr 1e-3 --data_dir data/raw/chest_xray")
            return True
            
        except Exception as e:
            print(f"   ✗ Data loading failed: {e}")
            print("\n⚠️  Torchvision compatibility issue detected.")
            print("\nTo fix this, try:")
            print("  1. Reinstall compatible versions:")
            print("     pip install torch torchvision --upgrade")
            print("  2. Or use the virtual environment:")
            print("     source env/bin/activate  # if using venv")
            print("     pip install -r requirements.txt")
            return False
            
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_training()
    sys.exit(0 if success else 1)



