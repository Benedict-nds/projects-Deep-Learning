#!/usr/bin/env python
"""
prepare_data.py
---------------
Preprocess and split chest X-ray dataset into train/val/test splits.
Fixes the imbalanced split issue (only 16 validation samples).
"""

import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_image_files(directory):
    """Get all image files from a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = []
    for ext in image_extensions:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))
    return files


def preprocess_and_split(
    raw_data_dir: str = "data/raw/chest_xray",
    processed_dir: str = "data/processed/chest_xray",
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    resize: bool = True,
    target_size: tuple = (224, 224),
    copy_files: bool = True
):
    """
    Preprocess and split dataset into train/val/test.
    
    Args:
        raw_data_dir: Directory with raw data (should have train/val/test or all images in class folders)
        processed_dir: Directory to save processed data
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
        seed: Random seed for reproducibility
        resize: Whether to resize images
        target_size: Target image size (width, height)
        copy_files: If True, copy files; if False, create symlinks
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    np.random.seed(seed)
    
    raw_path = Path(raw_data_dir)
    processed_path = Path(processed_dir)
    
    print("=" * 60)
    print("Data Preprocessing and Splitting")
    print("=" * 60)
    
    # Check if data is already split or needs splitting
    if (raw_path / "train").exists() and (raw_path / "val").exists() and (raw_path / "test").exists():
        print("\n Found existing train/val/test structure")
        print("   Will resplit to fix imbalanced validation set...")
        mode = "resplit"
    else:
        print("\n No existing splits found")
        print("   Will create new train/val/test splits...")
        mode = "new_split"
    
    # Get all classes
    if mode == "resplit":
        # Get classes from train directory
        train_path = raw_path / "train"
        classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    else:
        # Get classes from root directory
        classes = [d.name for d in raw_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n Found classes: {classes}")
    
    # Create processed directory structure
    for split in ["train", "val", "test"]:
        for class_name in classes:
            (processed_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    all_stats = {}
    
    for class_name in classes:
        print(f"\n Processing class: {class_name}")
        
        if mode == "resplit":
            # Collect all images from train, val, test
            all_images = []
            for split in ["train", "val", "test"]:
                split_dir = raw_path / split / class_name
                if split_dir.exists():
                    images = get_image_files(split_dir)
                    all_images.extend(images)
                    print(f"   Found {len(images)} images in {split}/")
        else:
            # Get all images from class directory
            class_dir = raw_path / class_name
            all_images = get_image_files(class_dir)
            print(f"   Found {len(all_images)} total images")
        
        if len(all_images) == 0:
            print(f"No images found for {class_name}, skipping...")
            continue
        
        # Shuffle
        np.random.shuffle(all_images)
        
        # Split: first train, then split remaining into val and test
        n_total = len(all_images)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val
        
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]
        
        print(f"   Split: Train={n_train}, Val={n_val}, Test={n_test}")
        
        # Process and copy images
        for split_name, images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            dest_dir = processed_path / split_name / class_name
            
            for img_path in tqdm(images, desc=f"   Processing {split_name}", leave=False):
                dest_path = dest_dir / img_path.name
                
                if resize:
                    try:
                        img = Image.open(img_path)
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Resize
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        img.save(dest_path, quality=95)
                    except Exception as e:
                        print(f"   Error processing {img_path}: {e}")
                        continue
                else:
                    if copy_files:
                        shutil.copy2(img_path, dest_path)
                    else:
                        # Create symlink
                        if dest_path.exists():
                            dest_path.unlink()
                        dest_path.symlink_to(os.path.abspath(img_path))
        
        all_stats[class_name] = {
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images),
            'total': n_total
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"\n{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    total_train = total_val = total_test = 0
    for class_name, stats in all_stats.items():
        print(f"{class_name:<15} {stats['train']:<10} {stats['val']:<10} {stats['test']:<10} {stats['total']:<10}")
        total_train += stats['train']
        total_val += stats['val']
        total_test += stats['test']
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_train:<10} {total_val:<10} {total_test:<10} {total_train + total_val + total_test:<10}")
    print("\n Preprocessing complete!")
    print(f" Processed data saved to: {processed_path}")
    #print(f"\n Use this directory for training:")
    #print(f"   python src/training/train.py --model cnn --epochs 10 --batch 32 --img_size 224 --lr 1e-4 --data_dir {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split chest X-ray dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw/chest_xray",
                        help="Directory with raw data")
    parser.add_argument("--processed_dir", type=str, default="data/processed/chest_xray",
                        help="Directory to save processed data")
    parser.add_argument("--train_split", type=float, default=0.7,
                        help="Proportion for training set")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Proportion for validation set")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Proportion for test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_resize", action="store_true",
                        help="Don't resize images (just copy)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Target image size")
    
    args = parser.parse_args()
    
    preprocess_and_split(
        raw_data_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        resize=not args.no_resize,
        target_size=(args.img_size, args.img_size),
        copy_files=True
    )


if __name__ == "__main__":
    main()



