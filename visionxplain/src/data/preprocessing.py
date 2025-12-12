# src/data/preprocessing.py
"""
Transforms for VisionXplain.
Designed for PyTorch + torchvision. 
ViT-ready default: 224x224, ImageNet normalization.
"""

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_transforms(img_size: int = 224):
    """
    Returns a dict with 'train', 'val', 'test' torchvision transforms.
    img_size: target image size (int)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # keep aspect ratio then center crop
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "val": val_test_transform,
        "test": val_test_transform
    }