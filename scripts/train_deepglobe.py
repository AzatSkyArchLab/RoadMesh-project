#!/usr/bin/env python3
"""
Train Road Segmentation Model on DeepGlobe Dataset

This script downloads the DeepGlobe Road Extraction dataset and trains
a segmentation model for road detection from satellite imagery.

Usage:
    python scripts/train_deepglobe.py --epochs 50 --batch_size 8

Requirements:
    - kaggle API token (~/.kaggle/kaggle.json)
    - GPU with 8+ GB VRAM recommended
"""
import argparse
import os
import sys
from pathlib import Path
import zipfile
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def download_deepglobe_dataset(data_dir: Path) -> Path:
    """Download DeepGlobe Road Extraction dataset from Kaggle."""
    dataset_path = data_dir / "deepglobe_roads"

    if dataset_path.exists() and (dataset_path / "train").exists():
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path

    print("Downloading DeepGlobe Road Extraction Dataset from Kaggle...")
    print("Make sure you have kaggle.json in ~/.kaggle/")

    try:
        import kaggle
        kaggle.api.authenticate()

        # Download dataset
        kaggle.api.dataset_download_files(
            "balraj98/deepglobe-road-extraction-dataset",
            path=str(data_dir),
            unzip=True
        )

        print(f"Dataset downloaded to {data_dir}")

        # Find the extracted folder
        for item in data_dir.iterdir():
            if item.is_dir() and "deepglobe" in item.name.lower():
                if item != dataset_path:
                    shutil.move(str(item), str(dataset_path))
                break

        return dataset_path

    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        print("\nAlternatively, download manually from:")
        print("https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset")
        print(f"Extract to: {dataset_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Kaggle account")
        print("2. API token at ~/.kaggle/kaggle.json")
        print("3. Accepted dataset terms on Kaggle website")
        sys.exit(1)


class DeepGlobeDataset(Dataset):
    """DeepGlobe Road Extraction Dataset."""

    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        transform=None,
        img_size: int = 512,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.img_size = img_size

        # Find images
        if split == "train":
            img_dir = self.root_dir / "train"
        else:
            img_dir = self.root_dir / "valid" if (self.root_dir / "valid").exists() else self.root_dir / "test"

        self.images = []
        self.masks = []

        # DeepGlobe format: {id}_sat.jpg and {id}_mask.png
        for img_path in sorted(img_dir.glob("*_sat.jpg")):
            mask_path = img_path.parent / img_path.name.replace("_sat.jpg", "_mask.png")
            if mask_path.exists():
                self.images.append(img_path)
                self.masks.append(mask_path)

        print(f"Found {len(self.images)} images in {split} split")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask) / 255.0).float()

        # Normalize image
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        # Add channel dim to mask
        mask = mask.unsqueeze(0)

        return image, mask


def train_model(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    img_size: int = 512,
):
    """Train segmentation model on DeepGlobe dataset."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = DeepGlobeDataset(data_dir, split="train", img_size=img_size)
    val_dataset = DeepGlobeDataset(data_dir, split="valid", img_size=img_size)

    if len(train_dataset) == 0:
        print("ERROR: No training images found!")
        print(f"Expected format: {data_dir}/train/*_sat.jpg and *_mask.png")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model using segmentation_models_pytorch
    try:
        import segmentation_models_pytorch as smp

        model = smp.Linknet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )
        print("Using LinkNet with ResNet34 encoder (ImageNet pretrained)")
    except ImportError:
        print("segmentation_models_pytorch not available, using custom model")
        from roadmesh.models.architectures import DLinkNet34
        model = DLinkNet34(num_classes=1, pretrained=True)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    # Training loop
    best_iou = 0.0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_iou = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Compute IoU
                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum() - intersection
                iou = (intersection + 1e-8) / (union + 1e-8)
                val_iou += iou.item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
            }
            torch.save(checkpoint, output_dir / "road_deepglobe_best.pt")
            print(f"  -> New best IoU: {best_iou:.4f}, saved checkpoint")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f"road_deepglobe_epoch{epoch+1}.pt")

    # Save final model
    torch.save(checkpoint, output_dir / "road_deepglobe_final.pt")

    # Copy best model to main checkpoints dir for use by server
    best_path = output_dir / "road_deepglobe_best.pt"
    target_path = Path("checkpoints/best_model.pt")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_path, target_path)

    print(f"\nTraining complete! Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {target_path}")
    print("\nYou can now use 'ML Model' mode in the web interface for road detection.")


def main():
    parser = argparse.ArgumentParser(description="Train road segmentation on DeepGlobe")
    parser.add_argument("--data_dir", type=str, default="data/deepglobe",
                       help="Directory for dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/deepglobe",
                       help="Directory for model outputs")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size (reduce if OOM)")
    parser.add_argument("--img_size", type=int, default=512,
                       help="Image size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--skip_download", action="store_true",
                       help="Skip dataset download (if already have data)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not args.skip_download:
        data_dir = download_deepglobe_dataset(data_dir)

    train_model(
        data_dir=data_dir,
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
