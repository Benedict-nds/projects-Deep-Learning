"""
predict.py
----------
Inference pipeline for VisionXplain models.
Loads trained models, makes predictions, and generates explainability visualizations.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.models.baseline.cnn import SimpleCNN
from src.models.vit.vit import VisionTransformer
from src.models.hybrid.cnn_vit import HybridNet
from src.data.preprocessing import get_transforms
from src.utils.helpers import denormalize_image, get_device
from src.explainability.gradcam.gradcam import GradCAM
from src.explainability.attention.attention import AttentionRollout


class ModelPredictor:
    """
    Unified predictor for VisionXplain models with explainability support.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        num_classes: int = 2,
        img_size: int = 224,
        class_names: Optional[list] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            model_type: 'cnn', 'vit', or 'hybrid'
            num_classes: Number of classes
            img_size: Image size
            class_names: List of class names
            device: Device to run on
        """
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.img_size = img_size
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        transforms = get_transforms(img_size=img_size)
        self.transform = transforms["test"]  # Use test transforms (no augmentation)
        
        # Initialize explainability methods
        self.gradcam = None
        self.attention_rollout = None
        
        if self.model_type == "cnn":
            self.gradcam = GradCAM(self.model)
        elif self.model_type in ["vit", "hybrid"]:
            self.attention_rollout = AttentionRollout(self.model)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        print(f"📦 Loading {self.model_type.upper()} model from: {model_path}")
        
        if self.model_type == "cnn":
            model = SimpleCNN(num_classes=self.num_classes, pretrained=False)
        elif self.model_type == "vit":
            model = VisionTransformer(
                img_size=self.img_size,
                patch_size=16,
                num_classes=self.num_classes,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072,
                dropout=0.1,
                return_attentions=False
            )
        elif self.model_type == "hybrid":
            model = HybridNet(
                num_classes=self.num_classes,
                cnn_backbone="resnet50",
                vit_config={
                    'img_size': self.img_size,
                    'patch_size': 16,
                    'dim': 768,
                    'depth': 12,
                    'heads': 12,
                    'mlp_dim': 3072,
                    'dropout': 0.1
                },
                fusion_method="concat"
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
            print(f"   ✓ Loaded checkpoint from epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✓ Loaded model state dict")
        
        return model
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (preprocessed_tensor, original_image_array)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        original_img = np.array(img)
        
        # Apply transforms
        img_tensor = self.transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor, original_img
    
    def predict(self, image_path: str) -> Dict:
        """
        Make prediction on an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        img_tensor, original_img = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = outputs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Get class probabilities
        class_probs = {
            self.class_names[i]: float(probs[0, i].item())
            for i in range(self.num_classes)
        }
        
        return {
            'predicted_class': self.class_names[pred_class],
            'predicted_class_idx': pred_class,
            'confidence': confidence,
            'class_probabilities': class_probs,
            'image_tensor': img_tensor,
            'original_image': original_img
        }
    
    def explain(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        target_class: Optional[int] = None
    ) -> Dict:
        """
        Generate prediction with explainability visualization.
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
            target_class: Optional target class for explanation (None = predicted class)
        
        Returns:
            Dictionary with prediction and explanation
        """
        # Get prediction
        result = self.predict(image_path)
        img_tensor = result['image_tensor']
        original_img = result['original_image']
        
        if target_class is None:
            target_class = result['predicted_class_idx']
        
        # Generate explanation
        explanation = {}
        
        if self.model_type == "cnn" and self.gradcam:
            # Grad-CAM for CNN
            try:
                heatmap_overlay = self.gradcam.generate_heatmap(
                    img_tensor,
                    original_img,
                    target_class=target_class
                )
                cam = self.gradcam.generate_cam(img_tensor, target_class=target_class)
                
                explanation['method'] = 'Grad-CAM'
                explanation['heatmap'] = cam
                explanation['overlay'] = heatmap_overlay
                
                # Visualize
                self._visualize_explanation(
                    original_img,
                    cam,
                    heatmap_overlay,
                    result,
                    save_path,
                    method='Grad-CAM'
                )
            except Exception as e:
                print(f"⚠️  Grad-CAM failed: {e}")
                explanation['error'] = str(e)
        
        elif self.model_type in ["vit", "hybrid"] and self.attention_rollout:
            # Attention Rollout for ViT/Hybrid
            try:
                attention_rollout, attention_to_cls = self.attention_rollout.generate_attention_map(
                    img_tensor,
                    img_size=self.img_size,
                    patch_size=16
                )
                
                # Reshape attention to image grid
                num_patches = int(np.sqrt(attention_to_cls.shape[0]))
                attention_map = attention_to_cls.reshape(num_patches, num_patches)
                
                # Resize to original image size
                import cv2
                h, w = original_img.shape[:2]
                attention_resized = cv2.resize(attention_map, (w, h))
                
                # Create overlay
                attention_colored = plt.cm.jet(attention_resized)[:, :, :3]
                overlay = (0.4 * attention_colored + 0.6 * original_img / 255.0)
                
                explanation['method'] = 'Attention Rollout'
                explanation['attention_map'] = attention_map
                explanation['overlay'] = overlay
                
                # Visualize
                self._visualize_explanation(
                    original_img,
                    attention_map,
                    overlay,
                    result,
                    save_path,
                    method='Attention Rollout'
                )
            except Exception as e:
                print(f"⚠️  Attention Rollout failed: {e}")
                explanation['error'] = str(e)
        
        result['explanation'] = explanation
        return result
    
    def _visualize_explanation(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        prediction: Dict,
        save_path: Optional[str],
        method: str
    ):
        """Visualize explanation results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img.astype(np.uint8))
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f'{method} Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        axes[2].imshow(overlay)
        pred_class = prediction['predicted_class']
        confidence = prediction['confidence']
        axes[2].set_title(
            f'Overlay\nPred: {pred_class} ({confidence:.1%})',
            fontsize=12,
            fontweight='bold'
        )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Explanation saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict and explain with VisionXplain models")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["cnn", "vit", "hybrid"], required=True,
                        help="Type of model")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--class_names", type=str, nargs='+', default=["NORMAL", "PNEUMONIA"],
                        help="Class names")
    parser.add_argument("--save_explanation", type=str, default=None,
                        help="Path to save explanation visualization")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class for explanation (None = predicted class)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VisionXplain Inference & Explainability")
    print("=" * 60)
    
    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"\n🖥️  Device: {device}")
    print(f"📷 Image: {args.image}")
    print(f"🤖 Model: {args.model_type.upper()}")
    
    # Create predictor
    predictor = ModelPredictor(
        model_path=args.model_path,
        model_type=args.model_type,
        num_classes=args.num_classes,
        img_size=args.img_size,
        class_names=args.class_names,
        device=device
    )
    
    # Predict and explain
    print(f"\n🔍 Running prediction and explanation...")
    result = predictor.explain(
        image_path=args.image,
        save_path=args.save_explanation,
        target_class=args.target_class
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nClass Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")
    
    if 'explanation' in result and 'method' in result['explanation']:
        print(f"\nExplanation Method: {result['explanation']['method']}")
        if args.save_explanation:
            print(f"Explanation saved to: {args.save_explanation}")
    
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()

