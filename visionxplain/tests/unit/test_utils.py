"""
Unit tests for utility functions.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import (
    set_seed,
    count_parameters,
    get_device,
    ensure_dir,
    denormalize_image,
    format_time
)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        val1 = torch.rand(1).item()
        
        set_seed(42)
        val2 = torch.rand(1).item()
        
        self.assertEqual(val1, val2, "Seed should produce reproducible results")
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        num_params = count_parameters(model)
        expected = 10 * 5 + 5  # weights + bias
        self.assertEqual(num_params, expected)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        # Should return CPU or CUDA
        self.assertIn(device.type, ['cpu', 'cuda'])
    
    def test_ensure_dir(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test" / "nested" / "dir"
            ensure_dir(str(test_dir))
            self.assertTrue(test_dir.exists())
            self.assertTrue(test_dir.is_dir())
    
    def test_denormalize_image(self):
        """Test image denormalization."""
        # Create normalized tensor (C, H, W)
        tensor = torch.rand(3, 224, 224) * 0.5 + 0.25  # Roughly normalized
        
        denorm = denormalize_image(tensor)
        
        self.assertEqual(denorm.shape, (224, 224, 3))
        self.assertTrue(denorm.dtype == np.uint8)
        self.assertTrue(denorm.min() >= 0)
        self.assertTrue(denorm.max() <= 255)
        
        # Test batch input
        batch_tensor = torch.rand(2, 3, 224, 224) * 0.5 + 0.25
        denorm_batch = denormalize_image(batch_tensor)
        self.assertEqual(denorm_batch.shape, (224, 224, 3))
    
    def test_format_time(self):
        """Test time formatting."""
        self.assertEqual(format_time(30), "30.00s")
        self.assertEqual(format_time(90), "1.50m")
        self.assertEqual(format_time(7200), "2.00h")


if __name__ == '__main__':
    unittest.main()

