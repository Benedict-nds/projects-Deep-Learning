"""
evaluate.py
-----------
Model evaluation utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import os
from pathlib import Path
from src.training.metrics import evaluate_model, MetricsCalculator


def evaluate_model_comprehensive(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[list] = None,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        class_names: List of class names
        save_dir: Optional directory to save results
    
    Returns:
        Dictionary containing evaluation results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Evaluate using metrics module
    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names,
        return_predictions=True
    )
    
    metrics = results['metrics']
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1-Score:  {metrics.get('f1', 0):.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("="*60)
    
    # Save results if directory provided
    if save_dir:
        import os
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        from src.utils.save_load import save_metrics
        save_metrics(metrics, os.path.join(save_dir, 'metrics.json'))
        
        # Generate and save visualizations
        metrics_calc = MetricsCalculator(
            num_classes=len(class_names) if class_names else 2,
            class_names=class_names
        )
        metrics_calc.all_predictions = results['predictions'].tolist()
        metrics_calc.all_labels = results['labels'].tolist()
        if 'probabilities' in results:
            metrics_calc.all_probs = results['probabilities'].tolist()
        
        # Confusion matrix
        metrics_calc.plot_confusion_matrix(
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # ROC curve
        if len(metrics_calc.all_probs) > 0:
            metrics_calc.plot_roc_curve(
                save_path=os.path.join(save_dir, 'roc_curve.png')
            )
        
        # Classification report
        metrics_calc.print_classification_report()
        metrics_calc.save_metrics(
            os.path.join(save_dir, 'classification_report.txt'),
            metrics
        )
    
    return results

