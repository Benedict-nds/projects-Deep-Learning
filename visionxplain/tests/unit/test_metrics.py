"""
Unit tests for metrics calculation.
"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.metrics import MetricsCalculator, compute_metrics_from_predictions


class TestMetrics(unittest.TestCase):
    """Test metrics calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_classes = 2
        self.class_names = ["NORMAL", "PNEUMONIA"]
        self.batch_size = 10
    
    def test_metrics_calculator_binary(self):
        """Test MetricsCalculator for binary classification."""
        calc = MetricsCalculator(num_classes=2, class_names=self.class_names)
        
        # Create perfect predictions
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        predictions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        probs = torch.softmax(torch.randn(10, 2), dim=1)
        
        calc.update(predictions, labels, probs)
        metrics = calc.compute()
        
        # Perfect predictions should have accuracy = 1.0
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)
        
        # Check per-class metrics exist
        self.assertIn('per_class_precision', metrics)
        self.assertIn('per_class_recall', metrics)
        self.assertIn('per_class_f1', metrics)
    
    def test_metrics_calculator_reset(self):
        """Test MetricsCalculator reset functionality."""
        calc = MetricsCalculator(num_classes=2)
        
        labels = torch.tensor([0, 1, 0, 1])
        predictions = torch.tensor([0, 1, 0, 1])
        probs = torch.softmax(torch.randn(4, 2), dim=1)
        
        calc.update(predictions, labels, probs)
        self.assertEqual(len(calc.all_predictions), 4)
        
        calc.reset()
        self.assertEqual(len(calc.all_predictions), 0)
        self.assertEqual(len(calc.all_labels), 0)
    
    def test_metrics_calculator_multiple_batches(self):
        """Test MetricsCalculator with multiple batches."""
        calc = MetricsCalculator(num_classes=2)
        
        # First batch
        labels1 = torch.tensor([0, 1])
        predictions1 = torch.tensor([0, 1])
        probs1 = torch.softmax(torch.randn(2, 2), dim=1)
        calc.update(predictions1, labels1, probs1)
        
        # Second batch
        labels2 = torch.tensor([0, 1])
        predictions2 = torch.tensor([0, 1])
        probs2 = torch.softmax(torch.randn(2, 2), dim=1)
        calc.update(predictions2, labels2, probs2)
        
        metrics = calc.compute()
        self.assertEqual(len(calc.all_predictions), 4)
        self.assertIn('accuracy', metrics)
    
    def test_compute_metrics_from_predictions(self):
        """Test compute_metrics_from_predictions function."""
        # Perfect predictions
        predictions = np.array([0, 1, 0, 1, 0, 1])
        labels = np.array([0, 1, 0, 1, 0, 1])
        probs = np.random.rand(6, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
        
        metrics = compute_metrics_from_predictions(
            predictions=predictions,
            labels=labels,
            probabilities=probs,
            class_names=self.class_names
        )
        
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertIn('roc_auc', metrics)
        self.assertIn('pr_auc', metrics)
    
    def test_metrics_with_wrong_predictions(self):
        """Test metrics with incorrect predictions."""
        calc = MetricsCalculator(num_classes=2)
        
        # All wrong predictions
        labels = torch.tensor([0, 1, 0, 1])
        predictions = torch.tensor([1, 0, 1, 0])  # All wrong
        probs = torch.softmax(torch.randn(4, 2), dim=1)
        
        calc.update(predictions, labels, probs)
        metrics = calc.compute()
        
        # Accuracy should be 0.0
        self.assertEqual(metrics['accuracy'], 0.0)
    
    def test_metrics_roc_auc(self):
        """Test ROC-AUC calculation."""
        calc = MetricsCalculator(num_classes=2)
        
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        predictions = torch.tensor([0, 1, 0, 1, 0, 1])
        # Create probabilities that favor correct class
        probs = torch.tensor([
            [0.9, 0.1],  # Correct
            [0.1, 0.9],  # Correct
            [0.8, 0.2],  # Correct
            [0.2, 0.8],  # Correct
            [0.7, 0.3],  # Correct
            [0.3, 0.7],  # Correct
        ])
        
        calc.update(predictions, labels, probs)
        metrics = calc.compute()
        
        self.assertIn('roc_auc', metrics)
        self.assertGreater(metrics['roc_auc'], 0.5)  # Should be good with these probs


if __name__ == '__main__':
    unittest.main()

