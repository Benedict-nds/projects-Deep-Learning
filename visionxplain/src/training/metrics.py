"""
metrics.py
----------
Comprehensive evaluation metrics for VisionXplain models.

This module provides:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve and AUC
- Class-wise evaluation
- Per-class metrics
- Integration with training loops
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """
    Comprehensive metrics calculator for classification tasks.
    Works with both binary and multi-class classification.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names for better reporting
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        # Storage for predictions and labels across batches
        self.reset()

    def reset(self):
        """Reset stored predictions and labels."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor, 
               probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: Predicted class indices (shape: [batch_size])
            labels: Ground truth labels (shape: [batch_size])
            probabilities: Predicted probabilities (shape: [batch_size, num_classes])
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()

        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)
        
        if probabilities is not None:
            self.all_probs.extend(probabilities)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions and labels.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if len(self.all_predictions) == 0:
            raise ValueError("No predictions accumulated. Call update() first.")

        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs) if len(self.all_probs) > 0 else None

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(
            labels, predictions, average='weighted', zero_division=0
        )
        metrics['recall'] = recall_score(
            labels, predictions, average='weighted', zero_division=0
        )
        metrics['f1'] = f1_score(
            labels, predictions, average='weighted', zero_division=0
        )

        # Per-class metrics
        if self.num_classes == 2:
            # Binary classification: also compute macro average
            metrics['precision_macro'] = precision_score(
                labels, predictions, average='macro', zero_division=0
            )
            metrics['recall_macro'] = recall_score(
                labels, predictions, average='macro', zero_division=0
            )
            metrics['f1_macro'] = f1_score(
                labels, predictions, average='macro', zero_division=0
            )

        # ROC AUC (requires probabilities)
        if probs is not None:
            if self.num_classes == 2:
                # Binary: use positive class probabilities
                metrics['roc_auc'] = roc_auc_score(labels, probs[:, 1])
                metrics['pr_auc'] = average_precision_score(labels, probs[:, 1])
            else:
                # Multi-class: use one-vs-rest
                metrics['roc_auc'] = roc_auc_score(
                    labels, probs, multi_class='ovr', average='weighted'
                )
                metrics['pr_auc'] = average_precision_score(
                    labels, probs, average='weighted'
                )

        # Per-class metrics
        per_class_metrics = self._compute_per_class_metrics(labels, predictions, probs)
        metrics.update(per_class_metrics)

        return metrics

    def _compute_per_class_metrics(self, labels: np.ndarray, predictions: np.ndarray,
                                   probs: Optional[np.ndarray]) -> Dict[str, Union[float, Dict]]:
        """Compute per-class metrics."""
        per_class = {}

        # Per-class precision, recall, F1
        precision_per_class = precision_score(
            labels, predictions, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            labels, predictions, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            labels, predictions, average=None, zero_division=0
        )

        per_class['per_class_precision'] = {
            self.class_names[i]: float(precision_per_class[i])
            for i in range(self.num_classes)
        }
        per_class['per_class_recall'] = {
            self.class_names[i]: float(recall_per_class[i])
            for i in range(self.num_classes)
        }
        per_class['per_class_f1'] = {
            self.class_names[i]: float(f1_per_class[i])
            for i in range(self.num_classes)
        }

        # Per-class ROC AUC (if probabilities available)
        if probs is not None:
            if self.num_classes == 2:
                # Binary: already computed in main metrics
                pass
            else:
                # Multi-class: compute per-class AUC
                per_class_auc = {}
                for i in range(self.num_classes):
                    # One-vs-rest for each class
                    y_binary = (labels == i).astype(int)
                    if len(np.unique(y_binary)) > 1:  # Check if class exists
                        per_class_auc[self.class_names[i]] = float(
                            roc_auc_score(y_binary, probs[:, i])
                        )
                per_class['per_class_roc_auc'] = per_class_auc

        return per_class

    def get_confusion_matrix(self) -> np.ndarray:
        """Compute and return confusion matrix."""
        if len(self.all_predictions) == 0:
            raise ValueError("No predictions accumulated.")
        
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        return confusion_matrix(labels, predictions, labels=list(range(self.num_classes)))

    def plot_confusion_matrix(self, save_path: Optional[str] = None, 
                             figsize: Tuple[int, int] = (8, 6)):
        """
        Plot confusion matrix as heatmap.
        
        Args:
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
        """
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_roc_curve(self, save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (8, 6)):
        """
        Plot ROC curve(s).
        
        Args:
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
        """
        if len(self.all_probs) == 0:
            raise ValueError("Probabilities not available. Cannot plot ROC curve.")
        
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        
        plt.figure(figsize=figsize)
        
        if self.num_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(labels, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        else:
            # Multi-class: plot one-vs-rest for each class
            for i in range(self.num_classes):
                y_binary = (labels == i).astype(int)
                if len(np.unique(y_binary)) > 1:
                    fpr, tpr, _ = roc_curve(y_binary, probs[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, 
                            label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

    def print_classification_report(self):
        """Print detailed classification report."""
        if len(self.all_predictions) == 0:
            raise ValueError("No predictions accumulated.")
        
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(
            labels, predictions, 
            target_names=self.class_names,
            digits=4
        ))
        print("="*60 + "\n")

    def save_metrics(self, save_path: str, metrics: Optional[Dict] = None):
        """
        Save metrics to a text file.
        
        Args:
            save_path: Path to save the metrics file
            metrics: Optional metrics dict (if None, computes from stored predictions)
        """
        if metrics is None:
            metrics = self.compute()
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION METRICS\n")
            f.write("="*60 + "\n\n")
            
            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write("-"*60 + "\n")
            f.write(f"Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score:  {metrics.get('f1', 0):.4f}\n")
            
            if 'roc_auc' in metrics:
                f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
            if 'pr_auc' in metrics:
                f.write(f"PR-AUC:    {metrics['pr_auc']:.4f}\n")
            
            f.write("\n")
            
            # Per-class metrics
            f.write("Per-Class Metrics:\n")
            f.write("-"*60 + "\n")
            
            if 'per_class_precision' in metrics:
                f.write("\nPrecision:\n")
                for class_name, value in metrics['per_class_precision'].items():
                    f.write(f"  {class_name}: {value:.4f}\n")
            
            if 'per_class_recall' in metrics:
                f.write("\nRecall:\n")
                for class_name, value in metrics['per_class_recall'].items():
                    f.write(f"  {class_name}: {value:.4f}\n")
            
            if 'per_class_f1' in metrics:
                f.write("\nF1-Score:\n")
                for class_name, value in metrics['per_class_f1'].items():
                    f.write(f"  {class_name}: {value:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"Metrics saved to {save_path}")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate a model on a dataloader and return comprehensive metrics.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: Optional list of class names
        return_predictions: If True, also return predictions and labels
        
    Returns:
        Dictionary containing metrics and optionally predictions/labels
    """
    model.eval()
    
    # Infer number of classes from dataloader if not provided
    if class_names is None:
        if hasattr(dataloader.dataset, 'classes'):
            class_names = dataloader.dataset.classes
        elif hasattr(dataloader.dataset, 'class_to_idx'):
            class_names = list(dataloader.dataset.class_to_idx.keys())
        else:
            # Try to infer from first batch
            first_batch = next(iter(dataloader))
            if isinstance(first_batch, (list, tuple)):
                labels = first_batch[1]
                num_classes = len(torch.unique(labels))
                class_names = [f"Class_{i}" for i in range(num_classes)]
            else:
                raise ValueError("Could not infer number of classes. Please provide class_names.")
    
    num_classes = len(class_names)
    metrics_calc = MetricsCalculator(num_classes=num_classes, class_names=class_names)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                images = batch
                labels = None
            
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions and probabilities
            if isinstance(outputs, dict):
                # Handle models that return dict (e.g., with attention)
                logits = outputs.get('logits', outputs.get('prediction', outputs))
            else:
                logits = outputs
            
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            if labels is not None:
                labels = labels.to(device)
                metrics_calc.update(predictions, labels, probs)
                
                if return_predictions:
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    metrics = metrics_calc.compute()
    
    result = {'metrics': metrics}
    
    if return_predictions:
        result['predictions'] = np.array(all_predictions)
        result['labels'] = np.array(all_labels)
        result['probabilities'] = np.array(all_probs)
    
    return result


def compute_metrics_from_predictions(
    predictions: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute metrics directly from predictions and labels.
    
    Args:
        predictions: Predicted class indices
        labels: Ground truth labels
        probabilities: Optional predicted probabilities
        class_names: Optional list of class names
        
    Returns:
        Dictionary containing computed metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if probabilities is not None and isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    
    # Infer number of classes
    num_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    metrics_calc = MetricsCalculator(num_classes=num_classes, class_names=class_names)
    metrics_calc.update(predictions, labels, probabilities)
    
    return metrics_calc.compute()

