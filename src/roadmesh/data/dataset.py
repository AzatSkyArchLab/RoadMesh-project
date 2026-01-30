"""
PyTorch Dataset and DataLoader

Handles loading and augmentation of training data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from roadmesh.core.config import AugmentationConfig


class RoadDataset(Dataset):
    """
    PyTorch Dataset for road segmentation.
    
    Expects directory structure:
    data_dir/
        images/
            tile_0.png
            tile_1.png
            ...
        masks/
            tile_0.png
            tile_1.png
            ...
    """
    
    def __init__(
        self,
        data_dir: Path,
        transform: Optional[A.Compose] = None,
        is_train: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing images/ and masks/ subdirs
            transform: Albumentations transform pipeline
            is_train: Whether this is training data (enables augmentation)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Find all images
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        # Get sorted list of image files
        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG images found in {self.images_dir}")
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.masks_dir / image_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Try different extensions
            for ext in ['.png', '.jpg', '.tif']:
                alt_path = mask_path.with_suffix(ext)
                if alt_path.exists():
                    mask = cv2.imread(str(alt_path), cv2.IMREAD_GRAYSCALE)
                    break
            else:
                # No mask found - create empty
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Normalize mask to [0, 1]
        if mask.max() > 1:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            # Default conversion to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()
        
        return image, mask


def create_train_transforms(config: Optional[AugmentationConfig] = None) -> A.Compose:
    """
    Create training augmentation pipeline.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Albumentations Compose transform
    """
    config = config or AugmentationConfig()
    
    transforms_list = []
    
    # Geometric augmentations
    if config.horizontal_flip:
        transforms_list.append(A.HorizontalFlip(p=0.5))
    
    if config.vertical_flip:
        transforms_list.append(A.VerticalFlip(p=0.5))
    
    if config.rotate_90:
        transforms_list.append(A.RandomRotate90(p=0.5))
    
    if config.rotate_limit > 0:
        transforms_list.append(
            A.Rotate(limit=config.rotate_limit, p=0.3, border_mode=cv2.BORDER_CONSTANT)
        )
    
    # Color augmentations
    if config.brightness_contrast:
        transforms_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=config.brightness_limit,
                contrast_limit=config.contrast_limit,
                p=0.5
            )
        )
    
    if config.blur_limit > 0:
        transforms_list.append(
            A.OneOf([
                A.GaussianBlur(blur_limit=config.blur_limit),
                A.MotionBlur(blur_limit=config.blur_limit),
            ], p=0.2)
        )
    
    if config.noise:
        transforms_list.append(
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(),
            ], p=0.2)
        )
    
    # Additional useful augmentations for satellite imagery
    transforms_list.extend([
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
    ])
    
    # Normalization and tensor conversion (always applied)
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def create_val_transforms() -> A.Compose:
    """
    Create validation transform pipeline (no augmentation).
    
    Returns:
        Albumentations Compose transform
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 4,
    num_workers: int = 4,
    augmentation_config: Optional[AugmentationConfig] = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Root data directory containing train/ and val/ subdirs
        batch_size: Batch size
        num_workers: Number of data loading workers
        augmentation_config: Augmentation settings
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = RoadDataset(
        data_dir / "train",
        transform=create_train_transforms(augmentation_config),
        is_train=True,
    )
    
    val_dataset = RoadDataset(
        data_dir / "val",
        transform=create_val_transforms(),
        is_train=False,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid batch size issues with BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    data_dir: Path,
    batch_size: int = 8,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create test data loader.
    
    Args:
        data_dir: Directory containing test/ subdir
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        Test DataLoader
    """
    data_dir = Path(data_dir)
    
    test_dataset = RoadDataset(
        data_dir / "test",
        transform=create_val_transforms(),
        is_train=False,
    )
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
