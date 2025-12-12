"""
cnn_vit.py
----------
Hybrid CNN-ViT model combining CNN features with Vision Transformer.
"""

import torch
import torch.nn as nn
from src.models.baseline.cnn import SimpleCNN
from src.models.vit.vit import VisionTransformer


class HybridNet(nn.Module):
    """
    Hybrid model combining CNN and ViT features.
    Uses CNN for local feature extraction and ViT for global attention.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        cnn_backbone: str = "resnet50",
        vit_config: dict = None,
        fusion_method: str = "concat"
    ):
        """
        Args:
            num_classes: Number of output classes
            cnn_backbone: CNN backbone architecture
            vit_config: Configuration dict for ViT (if None, uses defaults)
            fusion_method: How to fuse CNN and ViT features ('concat', 'add', 'attention')
        """
        super(HybridNet, self).__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # CNN branch
        self.cnn = SimpleCNN(num_classes=num_classes, backbone=cnn_backbone, pretrained=True)
        # CNN backbone outputs 2048 features (ResNet-50), we project to 512
        cnn_backbone_features = 2048 if cnn_backbone == "resnet50" else 1536  # EfficientNet-B3
        cnn_features = 512  # Target dimension after projection
        
        # ViT branch
        if vit_config is None:
            vit_config = {
                'img_size': 224,
                'patch_size': 16,
                'dim': 768,
                'depth': 12,
                'heads': 12,
                'mlp_dim': 3072,
                'dropout': 0.1
            }
        
        self.vit = VisionTransformer(
            num_classes=num_classes,
            return_attentions=False,
            **vit_config
        )
        # Remove classifier from ViT
        vit_features = vit_config['dim']
        
        # Fusion layer
        if fusion_method == "concat": 
            fused_features = cnn_features + vit_features  # 512 + 768 = 1280
        elif fusion_method == "add":
            # Ensure same dimension - project CNN to match ViT
            if cnn_features != vit_features:
                self.cnn_proj = nn.Linear(cnn_features, vit_features)
            fused_features = vit_features
        elif fusion_method == "attention":
            fused_features = cnn_features + vit_features
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=max(cnn_features, vit_features),
                num_heads=8
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Projection layers for feature extraction
        self.cnn_feat_proj = nn.Linear(cnn_backbone_features, cnn_features)
        # ViT already outputs 768-dim from CLS token, but add projection for safety
        self.vit_feat_proj = nn.Linear(vit_features, 768)  # Identity-like, ensures 768-dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fused_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Logits (B, num_classes)
        """
        # CNN branch - extract features before classifier
        cnn_feat = self.cnn.backbone(x)  # (B, 2048) for ResNet-50
        cnn_features = self.cnn_feat_proj(cnn_feat)  # Project to 512
        
        # ViT branch - extract CLS token features
        if hasattr(self.vit, 'patch_embed'):
            # Custom ViT: extract CLS token
            x_vit = self.vit.patch_embed(x)  # (B, N, dim)
            B = x_vit.shape[0]
            cls_tokens = self.vit.cls_token.expand(B, -1, -1)
            x_vit = torch.cat([cls_tokens, x_vit], dim=1)
            x_vit = x_vit + self.vit.pos_embed
            x_vit = self.vit.dropout(x_vit)
            
            for block in self.vit.blocks:
                x_vit, _ = block(x_vit)
            
            vit_features = self.vit.norm(x_vit)[:, 0]  # CLS token (B, dim)
        else:
            # Timm ViT or other: use forward hook or extract differently
            # For now, run forward and extract from intermediate
            vit_output = self.vit(x)
            # If output is logits, we need to extract features differently
            # This is a simplified approach - may need adjustment
            vit_features = vit_output  # Fallback
        
        # Ensure vit_features is right shape and project to 768
        if len(vit_features.shape) > 2:
            vit_features = vit_features.view(vit_features.size(0), -1)
        
        # Project to 768 (ViT should already be 768, but ensure it)
        vit_features = self.vit_feat_proj(vit_features)
        
        # Fusion
        if self.fusion_method == "add":
            if hasattr(self, 'cnn_proj'):
                cnn_features = self.cnn_proj(cnn_features)
            fused = cnn_features + vit_features
        
        elif self.fusion_method == "concat":
            fused = torch.cat([cnn_features, vit_features], dim=1)  # (B, 512+768)
        
        elif self.fusion_method == "attention":
            # Simplified: concatenate for now
            fused = torch.cat([cnn_features, vit_features], dim=1)
        
        # Final classification
        logits = self.classifier(fused)
        
        return logits


# for compatibility
HybridModel = HybridNet

