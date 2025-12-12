"""
gradcam.py
----------
Grad-CAM implementation for CNN models.
Generates heatmaps showing which regions of the image the model focuses on.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN models.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        """
        Args:
            model: PyTorch CNN model
            target_layer: Target convolutional layer to compute gradients for.
                         If None, will try to find the last convolutional layer.
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        if target_layer is None:
            self.target_layer = self._find_last_conv_layer()
        
        self._register_hooks()
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("No convolutional layer found in the model")
        
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        retain_graph: bool = False
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.
            retain_graph: Whether to retain computation graph
        
        Returns:
            Grad-CAM heatmap as numpy array (H, W)
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        score = output[0, target_class]
        score.backward(retain_graph=retain_graph)
        
        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()  # (C, H, W)
        activations = self.activations[0].cpu().data.numpy()  # (C, H, W)
        
        # Compute weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Generate heatmap overlay on original image.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            original_image: Original image as numpy array (H, W, 3) in range [0, 255]
            target_class: Target class index
            alpha: Transparency factor for overlay
        
        Returns:
            Overlayed image (H, W, 3) in range [0, 255]
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Resize CAM to match original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Convert to heatmap colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlayed
    
    def visualize(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        Visualize Grad-CAM results.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            original_image: Original image as numpy array (H, W, 3)
            target_class: Target class index
            save_path: Optional path to save figure
            figsize: Figure size
        """
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        if target_class is None:
            target_class = pred_class
        
        # Generate heatmap
        heatmap_overlay = self.generate_heatmap(input_tensor, original_image, target_class)
        cam = self.generate_cam(input_tensor, target_class)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(original_image.astype(np.uint8))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # CAM heatmap
        im = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(heatmap_overlay)
        axes[2].set_title(
            f'Overlay (Pred: {target_class}, Conf: {confidence:.3f})',
            fontsize=12,
            fontweight='bold'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def generate_gradcam_for_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_layer: Optional[torch.nn.Module] = None,
    device: torch.device = None
) -> np.ndarray:
    """
    Generate Grad-CAM for a batch of images.
    
    Args:
        model: PyTorch CNN model
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (B,)
        target_layer: Target layer for Grad-CAM
        device: Device to run on
    
    Returns:
        Array of CAM heatmaps (B, H, W)
    """
    if device is None:
        device = next(model.parameters()).device
    
    gradcam = GradCAM(model, target_layer)
    cams = []
    
    for i in range(images.shape[0]):
        img_tensor = images[i:i+1].to(device)
        cam = gradcam.generate_cam(img_tensor, target_class=labels[i].item())
        cams.append(cam)
    
    return np.array(cams)

