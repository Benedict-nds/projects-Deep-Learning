"""
vit.py
------
Vision Transformer (ViT) implementation for chest X-ray classification.
Supports both custom lightweight ViT and timm-based ViT models.
"""

import torch
import torch.nn as nn
import math
from typing import Optional
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Using custom ViT implementation.")


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, n_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x, attn_weights


class VisionTransformer(nn.Module):
    """
    Custom Vision Transformer implementation.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 2,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        return_attentions: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.return_attentions = return_attentions
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, dim)
        num_patches = self.patch_embed.n_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            logits: (B, num_classes)
            attentions: (optional) list of attention weights
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        attentions = []
        for block in self.blocks:
            x, attn = block(x)
            if self.return_attentions:
                attentions.append(attn)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # (B, dim)
        logits = self.head(cls_token_final)  # (B, num_classes)
        
        if self.return_attentions:
            return logits, attentions
        return logits


class ViTClassifier(nn.Module):
    """
    ViT Classifier using timm library (if available) or custom implementation.
    Provides a unified interface for ViT models.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1,
        return_attentions: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.return_attentions = return_attentions
        
        if TIMM_AVAILABLE and pretrained:
            # Use timm ViT
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout
            )
            self.use_timm = True
        else:
            # Use custom ViT
            # Parse model_name to extract config
            if "base" in model_name:
                dim, depth, heads, mlp_dim = 768, 12, 12, 3072
            elif "large" in model_name:
                dim, depth, heads, mlp_dim = 1024, 24, 16, 4096
            else:
                dim, depth, heads, mlp_dim = 768, 12, 12, 3072
            
            patch_size = 16 if "patch16" in model_name else 32
            img_size = 224 if "224" in model_name else 384
            
            self.model = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                num_classes=num_classes,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                return_attentions=return_attentions
            )
            self.use_timm = False
    
    def forward(self, x):
        """Forward pass."""
        if self.use_timm:
            return self.model(x)
        else:
            return self.model(x)
    
    def freeze_backbone(self):
        """Freeze all parameters except classification head."""
        for name, param in self.model.named_parameters():
            if 'head' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    def unfreeze_top_k_blocks(self, k: int = 2):
        """Unfreeze the last k transformer blocks (for custom ViT only)."""
        if not self.use_timm and hasattr(self.model, 'blocks'):
            total_blocks = len(self.model.blocks)
            for i in range(max(0, total_blocks - k), total_blocks):
                for param in self.model.blocks[i].parameters():
                    param.requires_grad = True

