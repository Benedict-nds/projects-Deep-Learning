#!/usr/bin/env python
"""
test_explainability.py
----------------------
Test explainability methods on multiple sample images.
Generates explanations for CNN, ViT, and Hybrid models.
"""

import argparse
import sys
import os
from pathlib import Path
from glob import glob

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predict import ModelPredictor


def test_explainability(
    model_paths: dict,
    image_paths: list,
    output_dir: str = "outputs/explanations",
    num_samples: int = 5
):
    """
    Test explainability on multiple images.
    
    Args:
        model_paths: Dict mapping model_type to model_path
        image_paths: List of image paths or directory to sample from
        output_dir: Directory to save explanations
        num_samples: Number of images to test per class
    """
    print("=" * 60)
    print("Explainability Testing")
    print("=" * 60)
    
    # Get sample images if directory provided
    if len(image_paths) == 1 and os.path.isdir(image_paths[0]):
        # Sample from directory
        all_images = []
        for ext in ['*.jpeg', '*.jpg', '*.png']:
            all_images.extend(glob(os.path.join(image_paths[0], '**', ext), recursive=True))
        
        # Sample evenly from classes if subdirectories exist
        sampled = []
        if any(os.path.isdir(os.path.join(image_paths[0], d)) for d in os.listdir(image_paths[0])):
            # Has class subdirectories
            for class_dir in os.listdir(image_paths[0]):
                class_path = os.path.join(image_paths[0], class_dir)
                if os.path.isdir(class_path):
                    class_images = []
                    for ext in ['*.jpeg', '*.jpg', '*.png']:
                        class_images.extend(glob(os.path.join(class_path, ext)))
                    sampled.extend(class_images[:num_samples])
        else:
            sampled = all_images[:num_samples * 2]
        
        image_paths = sampled
    
    print(f"\n📷 Testing on {len(image_paths)} images")
    print(f"💾 Saving to: {output_dir}")
    
    # Test each model
    results = {}
    
    for model_type, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"\n⚠️  Skipping {model_type}: {model_path} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing {model_type.upper()} Model")
        print(f"{'='*60}")
        
        try:
            # Create predictor
            predictor = ModelPredictor(
                model_path=model_path,
                model_type=model_type,
                num_classes=2,
                class_names=["NORMAL", "PNEUMONIA"]
            )
            
            model_results = []
            
            for i, img_path in enumerate(image_paths):
                print(f"\n[{i+1}/{len(image_paths)}] Processing: {os.path.basename(img_path)}")
                
                try:
                    # Generate explanation
                    save_path = os.path.join(
                        output_dir,
                        f"{model_type}_{os.path.basename(img_path).replace('.jpeg', '').replace('.jpg', '')}.png"
                    )
                    
                    result = predictor.explain(
                        image_path=img_path,
                        save_path=save_path,
                        target_class=None
                    )
                    
                    model_results.append({
                        'image': os.path.basename(img_path),
                        'predicted': result['predicted_class'],
                        'confidence': result['confidence'],
                        'explanation_saved': save_path
                    })
                    
                    print(f"   ✓ Predicted: {result['predicted_class']} ({result['confidence']:.1%})")
                    
                except Exception as e:
                    print(f"   ✗ Error: {e}")
                    continue
            
            results[model_type] = model_results
            
        except Exception as e:
            print(f"✗ Failed to load {model_type} model: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for model_type, model_results in results.items():
        print(f"\n{model_type.upper()} Model:")
        print(f"  Successfully processed: {len(model_results)} images")
        if model_results:
            avg_confidence = sum(r['confidence'] for r in model_results) / len(model_results)
            print(f"  Average confidence: {avg_confidence:.1%}")
    
    print(f"\n📁 All explanations saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test explainability on sample images")
    
    parser.add_argument("--cnn_model", type=str, default="outputs/models/best_cnn_model.pt",
                        help="Path to CNN model")
    parser.add_argument("--vit_model", type=str, default="outputs/models/best_vit_model.pt",
                        help="Path to ViT model")
    parser.add_argument("--hybrid_model", type=str, default="outputs/models/best_hybrid_model.pt",
                        help="Path to Hybrid model")
    parser.add_argument("--images", type=str, nargs='+',
                        help="Image paths or directory containing images")
    parser.add_argument("--data_dir", type=str, default="data/processed/chest_xray/test",
                        help="Directory to sample test images from")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of images per class to test")
    parser.add_argument("--output_dir", type=str, default="outputs/explanations",
                        help="Directory to save explanations")
    
    args = parser.parse_args()
    
    # Determine image paths
    if args.images:
        image_paths = args.images
    else:
        # Sample from test directory
        image_paths = [args.data_dir]
    
    # Build model paths dict
    model_paths = {}
    if os.path.exists(args.cnn_model):
        model_paths['cnn'] = args.cnn_model
    if os.path.exists(args.vit_model):
        model_paths['vit'] = args.vit_model
    if os.path.exists(args.hybrid_model):
        model_paths['hybrid'] = args.hybrid_model
    
    if not model_paths:
        print(" No model checkpoints found!")
        return
    
    # Run testing
    test_explainability(
        model_paths=model_paths,
        image_paths=image_paths,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()

