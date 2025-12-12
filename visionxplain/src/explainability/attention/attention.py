"""
attention.py
------------
ViT Attention Rollout for visualizing attention patterns in Vision Transformers.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pathlib import Path


class AttentionRollout:
    """
    Attention Rollout for Vision Transformers.
    Aggregates attention weights across layers to visualize what the model attends to.
    """
    
    def __init__(self, model: torch.nn.Module, head_fusion: str = "mean"):
        """
        Args:
            model: Vision Transformer model
            head_fusion: How to fuse attention heads ('mean', 'max', 'min')
        """
        self.model = model
        self.model.eval()
        self.head_fusion = head_fusion
        self.attention_weights = []
        
        # Register hooks to capture attention weights
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention weights from transformer blocks."""
        def hook_fn(module, input, output):
            # For custom ViT: output is tuple (x, attn_weights)
            # For timm ViT: need to extract from attention mechanism
            if isinstance(output, tuple):
                _, attn_weights = output
                if attn_weights is not None:
                    self.attention_weights.append(attn_weights.detach())
        
        # Try to register hooks on transformer blocks
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if hasattr(module, 'register_forward_hook'):
                    module.register_forward_hook(hook_fn)
    
    def rollout(
        self,
        attention_weights: List[torch.Tensor],
        discard_ratio: float = 0.9,
        head_fusion: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute attention rollout.
        
        Args:
            attention_weights: List of attention weight tensors from each layer
            discard_ratio: Ratio of attention to discard (keep top-k)
            head_fusion: How to fuse heads (overrides init value)
        
        Returns:
            Rolled out attention map (num_patches, num_patches)
        """
        if head_fusion is None:
            head_fusion = self.head_fusion
        
        # Process each layer's attention
        processed_attentions = []
        for attn in attention_weights:
            # attn shape: (batch, heads, seq_len, seq_len)
            if len(attn.shape) == 4:
                # Fuse heads
                if head_fusion == "mean":
                    attn = attn.mean(dim=1)
                elif head_fusion == "max":
                    attn = attn.max(dim=1)[0]
                elif head_fusion == "min":
                    attn = attn.min(dim=1)[0]
                else:
                    attn = attn.mean(dim=1)
            
            # Discard low attention values
            flat = attn.reshape(attn.shape[0], -1)
            _, indices = flat.topk(int(flat.shape[-1] * discard_ratio), dim=-1)
            indices = indices[indices != 0]  # Remove CLS token attention to itself
            flat[0, indices] = 0
            
            # Reshape back
            attn = flat.reshape(attn.shape)
            
            # Add identity for residual connection
            I = torch.eye(attn.shape[-1]).to(attn.device)
            attn = attn + I
            
            # Normalize
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            processed_attentions.append(attn[0].cpu().numpy())
        
        # Rollout: multiply attention matrices
        result = processed_attentions[0]
        for attn in processed_attentions[1:]:
            result = np.matmul(attn, result)
        
        return result
    
    def generate_attention_map(
        self,
        input_tensor: torch.Tensor,
        img_size: int = 224,
        patch_size: int = 16,
        discard_ratio: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate attention map for input image.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            img_size: Input image size
            patch_size: Patch size used in ViT
            discard_ratio: Ratio of attention to discard
        
        Returns:
            Tuple of (attention_map, attention_to_cls)
            - attention_map: (num_patches, num_patches) attention rollout
            - attention_to_cls: (num_patches,) attention to CLS token
        """
        self.attention_weights = []
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if len(self.attention_weights) == 0:
            # Try alternative method: extract from model if it returns attentions
            if hasattr(self.model, 'return_attentions'):
                # Temporarily enable attention return
                original_return = self.model.return_attentions
                self.model.return_attentions = True
                output, attentions = self.model(input_tensor)
                self.model.return_attentions = original_return
                
                if attentions:
                    self.attention_weights = attentions
        
        if len(self.attention_weights) == 0:
            raise ValueError("Could not extract attention weights from model")
        
        # Convert to list of tensors
        attn_list = []
        for attn in self.attention_weights:
            if isinstance(attn, torch.Tensor):
                attn_list.append(attn)
            elif isinstance(attn, list):
                attn_list.extend(attn)
        
        # Compute rollout
        attention_rollout = self.rollout(attn_list, discard_ratio=discard_ratio)
        
        # Extract attention to CLS token (first token)
        attention_to_cls = attention_rollout[0, 1:]  # Skip CLS token itself
        
        return attention_rollout, attention_to_cls
    
    def visualize_attention(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        img_size: int = 224,
        patch_size: int = 16,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Visualize attention rollout.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            original_image: Original image as numpy array (H, W, 3)
            img_size: Input image size
            patch_size: Patch size
            save_path: Optional path to save figure
            figsize: Figure size
        """
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Generate attention map
        attention_rollout, attention_to_cls = self.generate_attention_map(
            input_tensor, img_size, patch_size
        )
        
        # Reshape attention to image grid
        num_patches = int(np.sqrt(attention_to_cls.shape[0]))
        attention_map = attention_to_cls.reshape(num_patches, num_patches)
        
        # Resize to original image size
        import cv2
        h, w = original_image.shape[:2]
        attention_resized = cv2.resize(attention_map, (w, h))
        
        # Create overlay
        attention_colored = plt.cm.jet(attention_resized)[:, :, :3]
        overlay = (0.4 * attention_colored + 0.6 * original_image / 255.0)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(original_image.astype(np.uint8))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Attention map
        im = axes[1].imshow(attention_map, cmap='jet')
        axes[1].set_title('Attention Rollout', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(
            f'Overlay (Pred: {pred_class}, Conf: {confidence:.3f})',
            fontsize=12,
            fontweight='bold'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def visualize_vit_attention(
    model: torch.nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    img_size: int = 224,
    patch_size: int = 16,
    save_path: Optional[str] = None
):
    """
    Convenience function to visualize ViT attention.
    
    Args:
        model: ViT model
        image: Preprocessed image tensor (1, C, H, W)
        original_image: Original image array (H, W, 3)
        img_size: Image size
        patch_size: Patch size
        save_path: Optional save path
    """
    rollout = AttentionRollout(model)
    rollout.visualize_attention(
        image,
        original_image,
        img_size=img_size,
        patch_size=patch_size,
        save_path=save_path
    )

