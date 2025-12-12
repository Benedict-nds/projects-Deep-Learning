"""
cnn.py
------
Baseline CNN model for chest X-ray classification.
Uses ResNet-50 or EfficientNet-B3 as backbone with custom classification head.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    Simple CNN model using ResNet-50 backbone for chest X-ray classification.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, backbone: str = "resnet50"):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            backbone: Backbone architecture ('resnet50' or 'efficientnet_b3')
        """
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            # Replace the final fully connected layer
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        elif backbone == "efficientnet_b3":
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# Alias for compatibility
BaselineCNN = SimpleCNN

