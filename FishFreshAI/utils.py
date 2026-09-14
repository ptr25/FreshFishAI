"""Utility functions for the FishFresh AI project."""

import os
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """Return train/val test torchvision transforms.

    - Training: augmentation + normalization
    - Validation / Testing: only resize + normalization
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return {"train": train_transform, "val": val_transform, "test": val_transform}


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[str, Any]]:
    """Create train, validation and test DataLoaders.

    The dataset directory is expected to contain sub-folders:
        data_dir/
            fresh/
            non_fresh/
    (Standard ImageFolder layout; binary classification.)

    Returns:
        loaders: dict with keys "train", "val", "test"
        dataset_sizes: dict with numbers of images per split
        image_datasets: dict of ImageFolder datasets
    """
    transforms_dict = get_transforms(img_size)

    image_datasets = {
        x: datasets.ImageFolder(
            root=os.path.join(data_dir, x), transform=transforms_dict[x]
        )
        for x in ["train", "val", "test"]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True if x == "train" else False,
            num_workers=num_workers,
        )
        for x in ["train", "val", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

    return dataloaders, dataset_sizes, image_datasets


def compute_class_weights(
    dataset, device: torch.device
) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.

    pos_weight = number_of_negatives / number_of_positives
    """
    counts = {}
    for _, label in dataset:
        # label is 0 or 1 (from ImageFolder)
        counts[label] = counts.get(label, 0) + 1

    n_pos = counts.get(1, 0)
    n_neg = counts.get(0, 0)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Dataset must contain both classes.")

    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    return pos_weight