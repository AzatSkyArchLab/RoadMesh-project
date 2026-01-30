#!/usr/bin/env python
"""
Training Script

Train road segmentation model with configurable parameters.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/model/dlinknet_8gb.yaml
    python scripts/train.py --epochs 50 --batch-size 2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train RoadMesh model")
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/dlinknet_8gb.yaml"),
        help="Path to config file"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed dataset"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save checkpoints"
    )
    
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint"
    )
    
    # Override config values
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--architecture", type=str, default=None)
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for logging"
    )
    
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (1 epoch, small subset)"
    )
    
    return parser.parse_args()


def check_gpu():
    """Check GPU availability and print info."""
    print("\n" + "="*60)
    print("GPU Information")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Training will be slow on CPU.")
        return "cpu"
    
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    print(f"Device: {gpu_name}")
    print(f"Total VRAM: {total_memory:.1f} GB")
    
    # Check for RTX 3070 Ti or similar
    if total_memory < 6:
        print("⚠ Less than 6GB VRAM. Consider reducing batch_size to 2.")
    elif total_memory < 10:
        print("✓ 6-10GB VRAM. Using batch_size=4 with mixed precision.")
    else:
        print("✓ 10+ GB VRAM. Can use larger batch sizes.")
    
    return f"cuda:{device}"


def check_dataset(data_dir: Path):
    """Verify dataset exists and has expected structure."""
    print("\n" + "="*60)
    print("Dataset Check")
    print("="*60)
    
    required_dirs = [
        data_dir / "train" / "images",
        data_dir / "train" / "masks",
        data_dir / "val" / "images",
        data_dir / "val" / "masks",
    ]
    
    for d in required_dirs:
        if not d.exists():
            print(f"✗ Missing: {d}")
            print("\nPlease run prepare_dataset.py first:")
            print("  python scripts/prepare_dataset.py --help")
            sys.exit(1)
        
        n_files = len(list(d.glob("*.png")))
        print(f"✓ {d}: {n_files} files")
    
    train_count = len(list((data_dir / "train" / "images").glob("*.png")))
    val_count = len(list((data_dir / "val" / "images").glob("*.png")))
    
    print(f"\nTotal: {train_count} train, {val_count} val")
    
    if train_count < 10:
        print("⚠ Very small dataset. Results may be poor.")
    
    return train_count, val_count


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("ROADMESH TRAINING")
    print("="*60)
    
    # Check GPU
    device = check_gpu()
    
    # Check dataset
    train_count, val_count = check_dataset(args.data_dir)
    
    # Load config
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    
    from roadmesh.core.config import RoadMeshConfig
    
    if args.config.exists():
        config = RoadMeshConfig.from_yaml(args.config)
        print(f"Loaded config: {args.config}")
    else:
        config = RoadMeshConfig()
        print("Using default config")
    
    # Apply command line overrides
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.architecture:
        config.model.architecture = args.architecture
    if args.no_mixed_precision:
        config.training.mixed_precision = False
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir
    
    if args.debug:
        config.training.epochs = 1
        config.training.batch_size = 2
        print("⚠ Debug mode: 1 epoch, batch_size=2")
    
    print(f"\nSettings:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation}")
    print(f"  Effective batch size: {config.training.batch_size * config.training.gradient_accumulation}")
    print(f"  Mixed precision: {config.training.mixed_precision}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.epochs}")
    
    # Create model
    print("\n" + "="*60)
    print("Model")
    print("="*60)
    
    from roadmesh.models.architectures import create_model, count_parameters, estimate_memory_usage
    
    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained_backbone,
        device=device,
    )
    
    params = count_parameters(model)
    print(f"Parameters: {params['total']:,} ({params['total_mb']:.1f} MB)")
    
    memory = estimate_memory_usage(
        model,
        batch_size=config.training.batch_size,
        mixed_precision=config.training.mixed_precision,
    )
    print(f"Estimated VRAM: {memory['total_gb']:.1f} GB")
    
    if not memory['fits_8gb']:
        print("⚠ May not fit in 8GB VRAM. Consider reducing batch_size.")
    
    # Create data loaders
    print("\n" + "="*60)
    print("Data Loaders")
    print("="*60)
    
    from roadmesh.data.dataset import create_dataloaders
    
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        augmentation_config=config.data.augmentations,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    from roadmesh.training.trainer import Trainer
    
    trainer = Trainer(
        model=model,
        config=config.training,
        device=device,
        experiment_name=args.experiment_name,
    )
    
    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train!
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    try:
        history = trainer.fit(train_loader, val_loader)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best IoU: {trainer.best_metric:.4f}")
        print(f"Checkpoints saved to: {config.training.checkpoint_dir}")
        print(f"\nTo run inference:")
        print(f"  python scripts/predict.py --checkpoint {config.training.checkpoint_dir}/best_model.pt")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted.pt")
        print(f"Saved to: {config.training.checkpoint_dir}/interrupted.pt")
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "="*60)
            print("OUT OF MEMORY ERROR")
            print("="*60)
            print("GPU ran out of memory. Try:")
            print("  1. Reduce batch_size: --batch-size 2")
            print("  2. Enable gradient checkpointing in config")
            print("  3. Close other GPU applications")
            torch.cuda.empty_cache()
        raise


if __name__ == "__main__":
    main()
