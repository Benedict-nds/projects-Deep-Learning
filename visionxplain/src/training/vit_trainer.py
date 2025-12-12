"""
vit_trainer.py
---------------
ViT-specific trainer with support for attention visualization and metrics.
"""

import torch
import torch.nn as nn
import os
from typing import Optional
from tqdm import tqdm
from src.training.metrics import MetricsCalculator


class ViTTrainer:
    """
    Trainer for Vision Transformer models.
    Includes metrics tracking and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lr: float = 3e-5,
        device: Optional[torch.device] = None,
        save_dir: str = "outputs/models"
    ):
        """
        Args:
            model: ViT model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            device: Device to train on (auto-detected if None)
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_dir = save_dir
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Get class names from dataloader
        if hasattr(train_loader.dataset, 'classes'):
            self.class_names = train_loader.dataset.classes
        elif hasattr(train_loader.dataset, 'class_to_idx'):
            self.class_names = list(train_loader.dataset.class_to_idx.keys())
        else:
            self.class_names = None
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return running_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        
        metrics_calc = None
        if self.class_names:
            num_classes = len(self.class_names)
            metrics_calc = MetricsCalculator(num_classes=num_classes, class_names=self.class_names)
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                if metrics_calc:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    metrics_calc.update(preds, labels, probs)
        
        avg_loss = running_loss / len(self.val_loader)
        
        metrics = {}
        if metrics_calc:
            metrics = metrics_calc.compute()
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'best_vit_model.pt')
        else:
            path = os.path.join(self.save_dir, f'vit_checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        })
        
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"   Resuming from epoch {start_epoch + 1}")
        
        return start_epoch
    
    def fit(self, early_stopping_patience: int = 5, resume_from: str = None):
        """
        Main training loop.
        
        Args:
            early_stopping_patience: Number of epochs to wait before stopping if no improvement
            resume_from: Path to checkpoint to resume from
        """
        print(f"\n🚀 Starting ViT Training")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        if early_stopping_patience > 0:
            print(f"Early stopping patience: {early_stopping_patience}")
        print("-" * 60)
        
        start_epoch = 0
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
            # Restore best values from history
            if self.history.get('val_loss'):
                best_val_loss = min(self.history['val_loss'])
            if self.history.get('val_accuracy'):
                best_val_acc = max(self.history['val_accuracy'])
        
        for epoch in range(start_epoch + 1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.history['val_loss'].append(val_loss)
            
            if val_metrics:
                val_acc = val_metrics.get('accuracy', 0)
                self.history['val_accuracy'].append(val_acc)
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
                # Check for improvement (using accuracy if available, else loss)
                improved = val_acc > best_val_acc if val_acc > 0 else val_loss < best_val_loss
                
                if improved:
                    if val_acc > 0:
                        best_val_acc = val_acc
                    best_val_loss = val_loss
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                    is_best = False
                    
                # Early stopping check
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                    print(f"Best validation accuracy: {best_val_acc:.4f}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            else:
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    patience_counter = 0
                    is_best = True
                else:
                    patience_counter += 1
                    is_best = False
                
                print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            
            # Learning rate step
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print("\n✅ Training completed!")
        return self.history

