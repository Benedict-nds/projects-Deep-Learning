# src/data/dataloader.py

import os
from torchvision import datasets
from torch.utils.data import DataLoader
from src.data.preprocessing import get_transforms

def create_dataloaders(
    data_root: str = "data/raw/chest_xray",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Returns: dict with train/val/test DataLoaders and class_to_idx mapping.
    """
    transforms = get_transforms(img_size)

    subsets = {}
    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Expected dataset folder at: {split_dir}")
        subsets[split] = datasets.ImageFolder(root=split_dir, transform=transforms[split])

    # Sanity: classes and counts
    class_to_idx = subsets["train"].class_to_idx
    counts = {s: len(subsets[s]) for s in subsets}

    dataloaders = {
        "train": DataLoader(
            subsets["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "val": DataLoader(
            subsets["val"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "test": DataLoader(
            subsets["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
    }

    meta = {
        "class_to_idx": class_to_idx,
        "counts": counts
    }

    return dataloaders, meta

if __name__ == "__main__":
    # Quick local sanity check (run python -m src.data.dataloader from repo root)
    import pprint
    dl, meta = create_dataloaders(data_root="data/raw/chest_xray", img_size=224, batch_size=8, num_workers=2)
    pprint.pprint(meta)
    # iterate one batch
    batch = next(iter(dl["train"]))
    images, labels = batch
    print("Batch shapes:", images.shape, labels.shape)