"""
Integration tests for training pipeline.
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path
import tempfile
import shutil

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.baseline.cnn import SimpleCNN
from src.models.vit.vit import VisionTransformer
from src.training.finetune import CNNTrainer
from src.training.vit_trainer import ViTTrainer
from src.utils.helpers import set_seed


class TestTraining(unittest.TestCase):
    """Test training pipeline integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        set_seed(42)
        self.batch_size = 4
        self.img_size = 224
        self.num_classes = 2
        self.device = torch.device("cpu")
        
        # Create dummy dataset
        self.train_data = torch.utils.data.TensorDataset(
            torch.randn(20, 3, self.img_size, self.img_size),
            torch.randint(0, self.num_classes, (20,))
        )
        self.val_data = torch.utils.data.TensorDataset(
            torch.randn(10, 3, self.img_size, self.img_size),
            torch.randint(0, self.num_classes, (10,))
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )
        
        # Temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cnn_trainer_initialization(self):
        """Test CNN trainer initialization."""
        model = SimpleCNN(num_classes=self.num_classes, pretrained=False)
        trainer = CNNTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1,
            lr=1e-4,
            device=self.device,
            save_dir=self.temp_dir
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)
    
    def test_cnn_trainer_train_epoch(self):
        """Test CNN trainer training one epoch."""
        model = SimpleCNN(num_classes=self.num_classes, pretrained=False)
        trainer = CNNTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1,
            lr=1e-4,
            device=self.device,
            save_dir=self.temp_dir
        )
        
        # Train one epoch
        loss = trainer.train_epoch()
        
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
    
    def test_cnn_trainer_validation(self):
        """Test CNN trainer validation."""
        model = SimpleCNN(num_classes=self.num_classes, pretrained=False)
        trainer = CNNTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1,
            lr=1e-4,
            device=self.device,
            save_dir=self.temp_dir
        )
        
        # Validate (returns tuple of loss and metrics)
        val_loss, metrics = trainer.validate()
        
        self.assertIsInstance(val_loss, float)
        self.assertGreaterEqual(val_loss, 0)
        self.assertIsInstance(metrics, dict)
    
    def test_cnn_trainer_checkpoint_saving(self):
        """Test CNN trainer checkpoint saving."""
        model = SimpleCNN(num_classes=self.num_classes, pretrained=False)
        trainer = CNNTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1,
            lr=1e-4,
            device=self.device,
            save_dir=self.temp_dir
        )
        
        # Train one epoch to populate history
        train_loss = trainer.train_epoch()
        trainer.history['train_loss'].append(train_loss)
        
        # Validate to populate validation history
        val_loss, val_metrics = trainer.validate()
        trainer.history['val_loss'].append(val_loss)
        if val_metrics:
            trainer.history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
        
        # Save checkpoint
        trainer.save_checkpoint(epoch=0, is_best=True)
        
        # Check if checkpoint file exists
        checkpoint_path = Path(self.temp_dir) / "best_cnn_model.pt"
        self.assertTrue(checkpoint_path.exists())
        
        # Create a new trainer to test loading
        model2 = SimpleCNN(num_classes=self.num_classes, pretrained=False)
        trainer2 = CNNTrainer(
            model=model2,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1,
            lr=1e-4,
            device=self.device,
            save_dir=self.temp_dir
        )
        
        # Test loading
        start_epoch = trainer2.load_checkpoint(str(checkpoint_path))
        
        # Check that checkpoint was loaded
        self.assertEqual(start_epoch, 0)
        self.assertEqual(len(trainer2.history['train_loss']), 1)
        self.assertEqual(len(trainer2.history['val_loss']), 1)
        self.assertEqual(len(trainer2.history['val_accuracy']), 1)
    
    def test_vit_trainer_initialization(self):
        """Test ViT trainer initialization."""
        model = VisionTransformer(
            img_size=self.img_size,
            patch_size=16,
            num_classes=self.num_classes,
            dim=384,  # Smaller for testing
            depth=4,
            heads=6,
            mlp_dim=1536
        )
        trainer = ViTTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1,
            lr=3e-5,
            device=self.device,
            save_dir=self.temp_dir
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)
    
    def test_vit_trainer_train_epoch(self):
        """Test ViT trainer training one epoch."""
        model = VisionTransformer(
            img_size=self.img_size,
            patch_size=16,
            num_classes=self.num_classes,
            dim=384,
            depth=4,
            heads=6,
            mlp_dim=1536
        )
        trainer = ViTTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1,
            lr=3e-5,
            device=self.device,
            save_dir=self.temp_dir
        )
        
        # Train one epoch
        loss = trainer.train_epoch()
        
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)


if __name__ == '__main__':
    unittest.main()

