import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import random
import numpy as np
from typing import Optional, Tuple, List


def get_imagenet_subset_indices(dataset: Dataset, num_classes: int, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Select balanced subset of ImageNet classes."""
    random.seed(seed)
    np.random.seed(seed)
    
    all_targets = np.array([dataset.targets[i] if hasattr(dataset, 'targets') else dataset[i][1] 
                           for i in range(len(dataset))])
    unique_classes = np.unique(all_targets)
    
    if num_classes > len(unique_classes):
        raise ValueError(f"Requested {num_classes} classes but dataset has only {len(unique_classes)}")
    
    selected_classes = sorted(random.sample(list(unique_classes), num_classes))
    
    class_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_classes)}
    
    subset_indices = []
    new_targets = []
    for idx in range(len(dataset)):
        target = all_targets[idx]
        if target in selected_classes:
            subset_indices.append(idx)
            new_targets.append(class_to_new_idx[target])
    
    return subset_indices, new_targets


class ImageNetSubset(Dataset):
    """Wrapper for ImageNet subset with remapped labels."""
    
    def __init__(self, base_dataset: Dataset, indices: List[int], targets: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices
        self.targets = targets
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[self.indices[idx]]
        target = self.targets[idx]
        return img, target


def get_train_transform(input_size: int = 224, auto_augment: bool = True):
    """Training data augmentation pipeline."""
    transforms_list = [
        transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    
    if auto_augment:
        transforms_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    else:
        transforms_list.extend([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
    
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    
    return transforms.Compose(transforms_list)


def get_val_transform(input_size: int = 224):
    """Validation data preprocessing."""
    resize_size = int(input_size / 0.875)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_dataset(
    data_path: str,
    subset_classes: Optional[int] = None,
    input_size: int = 224,
    auto_augment: bool = True,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, int]:
    """Build train and validation datasets."""
    
    data_path = Path(data_path)
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        raise ValueError(f"ImageNet data not found at {data_path}. Expected 'train' and 'val' subdirectories.")
    
    train_transform = get_train_transform(input_size, auto_augment)
    val_transform = get_val_transform(input_size)
    
    base_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    base_val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    if subset_classes is not None:
        train_indices, train_targets = get_imagenet_subset_indices(base_train_dataset, subset_classes, seed)
        val_indices, val_targets = get_imagenet_subset_indices(base_val_dataset, subset_classes, seed)
        
        train_dataset = ImageNetSubset(base_train_dataset, train_indices, train_targets)
        val_dataset = ImageNetSubset(base_val_dataset, val_indices, val_targets)
        num_classes = subset_classes
    else:
        train_dataset = base_train_dataset
        val_dataset = base_val_dataset
        num_classes = len(base_train_dataset.classes)
    
    return train_dataset, val_dataset, num_classes


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Build dataloader with optimized settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
