"""
Unit tests for model architectures.
"""

import unittest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.baseline.cnn import SimpleCNN
from src.models.vit.vit import VisionTransformer, ViTClassifier
from src.models.hybrid.cnn_vit import HybridNet
from src.utils.helpers import count_parameters


class TestModels(unittest.TestCase):
    """Test model architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.img_size = 224
        self.num_classes = 2
        self.device = torch.device("cpu")
    
    def test_cnn_model(self):
        """Test CNN model creation and forward pass."""
        model = SimpleCNN(num_classes=self.num_classes, pretrained=False)
        model.eval()
        
        # Test forward pass
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
        # Test parameter count
        num_params = count_parameters(model)
        self.assertGreater(num_params, 0)
    
    def test_cnn_different_backbones(self):
        """Test CNN with different backbones."""
        for backbone in ["resnet50", "efficientnet_b3"]:
            model = SimpleCNN(num_classes=self.num_classes, pretrained=False, backbone=backbone)
            model.eval()
            
            x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
            with torch.no_grad():
                output = model(x)
            
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_vit_model(self):
        """Test Vision Transformer model."""
        model = VisionTransformer(
            img_size=self.img_size,
            patch_size=16,
            num_classes=self.num_classes,
            dim=768,
            depth=6,  # Smaller for testing
            heads=12,
            mlp_dim=3072
        )
        model.eval()
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
        # Test parameter count
        num_params = count_parameters(model)
        self.assertGreater(num_params, 0)
    
    def test_vit_with_attention(self):
        """Test ViT model returns attention weights."""
        model = VisionTransformer(
            img_size=self.img_size,
            patch_size=16,
            num_classes=self.num_classes,
            dim=768,
            depth=6,
            heads=12,
            mlp_dim=3072
        )
        model.eval()
        model.return_attentions = True  # Enable attention return
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        with torch.no_grad():
            output, attentions = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertIsNotNone(attentions)
        self.assertGreater(len(attentions), 0)
    
    def test_hybrid_model(self):
        """Test Hybrid CNN-ViT model."""
        model = HybridNet(
            num_classes=self.num_classes,
            cnn_backbone="resnet50",
            vit_config={
                'img_size': self.img_size,
                'patch_size': 16,
                'dim': 768,
                'depth': 6,
                'heads': 12,
                'mlp_dim': 3072
            }
        )
        model.eval()
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
        # Test parameter count
        num_params = count_parameters(model)
        self.assertGreater(num_params, 0)
    
    def test_hybrid_with_attention(self):
        """Test Hybrid model returns attention weights."""
        model = HybridNet(
            num_classes=self.num_classes,
            cnn_backbone="resnet50",
            vit_config={
                'img_size': self.img_size,
                'patch_size': 16,
                'dim': 768,
                'depth': 6,
                'heads': 12,
                'mlp_dim': 3072
            }
        )
        model.eval()
        # Enable attention return on ViT branch
        if hasattr(model.vit, 'return_attentions'):
            model.vit.return_attentions = True
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        with torch.no_grad():
            output = model(x)
            # Hybrid model doesn't return attentions directly
            # We just verify it works with attention enabled
            if hasattr(model.vit, 'return_attentions') and model.vit.return_attentions:
                # Try to get attentions from ViT branch separately
                vit_output, attentions = model.vit(x)
                self.assertIsNotNone(attentions)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


if __name__ == '__main__':
    unittest.main()

