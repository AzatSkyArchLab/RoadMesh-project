"""
Pretrained Road Segmentation Models

Provides easy access to pretrained models for road extraction from satellite imagery.
"""
from __future__ import annotations

import os
import urllib.error
from pathlib import Path
from typing import Optional, Literal
import urllib.request

import torch
import torch.nn as nn

# Try to import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

# Try to import huggingface_hub for model downloads
try:
    from huggingface_hub import hf_hub_download, HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# HuggingFace models for satellite/road segmentation
HF_MODELS = {
    "satellite_segmentation": {
        "repo_id": "abcd1334/Satellite_Segmentation",
        "description": "Multi-class satellite segmentation (buildings, roads, vegetation, water)",
    },
    "aerial_roads_unet": {
        "repo_id": "spectrewolf8/aerial-image-road-segmentation-with-U-NET-xp",
        "description": "U-Net trained on Massachusetts Roads Dataset",
    },
}


# URLs for pretrained weights (direct downloads)
PRETRAINED_URLS = {
    # D-LinkNet trained on DeepGlobe (from original authors)
    "dlinknet_deepglobe": "https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=1",
}

# Known checkpoint filenames to search for
KNOWN_CHECKPOINT_NAMES = [
    "dlinknet34_deepglobe.pt",
    "dlinknet34_deepglobe.pth",
    "dlinknet34_deepglobe.th",
    "dlink34_road.pt",
    "road_deepglobe_best.pt",
    "best_model.pt",
    "road_finetuned.pt",
]


class PretrainedRoadSegmentation(nn.Module):
    """
    Road segmentation model with pretrained ImageNet encoder.

    Uses segmentation_models_pytorch for quick setup with strong encoders.
    The encoder is pretrained on ImageNet, giving better feature extraction
    than random initialization.
    """

    def __init__(
        self,
        architecture: Literal["unet", "linknet", "fpn", "pspnet", "deeplabv3plus"] = "linknet",
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        num_classes: int = 1,
    ):
        super().__init__()

        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation_models_pytorch is required. "
                "Install with: pip install segmentation-models-pytorch"
            )

        self.architecture = architecture
        self.encoder_name = encoder_name

        # Create model based on architecture
        model_class = {
            "unet": smp.Unet,
            "linknet": smp.Linknet,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "deeplabv3plus": smp.DeepLabV3Plus,
        }.get(architecture)

        if model_class is None:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Try to create model with pretrained weights, fall back to no weights if network fails
        try:
            self.model = model_class(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
                activation=None,  # We'll apply sigmoid ourselves
            )
        except (OSError, urllib.error.URLError) as e:
            print(f"Warning: Could not download pretrained weights ({e})")
            print("Creating model without pretrained encoder weights")
            self.model = model_class(
                encoder_name=encoder_name,
                encoder_weights=None,  # No pretrained weights
                in_channels=3,
                classes=num_classes,
                activation=None,
            )

        # Get preprocessing params for this encoder
        self.preprocess_params = smp.encoders.get_preprocessing_params(
            encoder_name, pretrained=encoder_weights
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_preprocessing_fn(self):
        """Get preprocessing function for this model's encoder."""
        mean = self.preprocess_params["mean"]
        std = self.preprocess_params["std"]

        def preprocess(img):
            # img should be [B, C, H, W] tensor normalized to [0, 1]
            mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(img.device)
            std_t = torch.tensor(std).view(1, 3, 1, 1).to(img.device)
            return (img - mean_t) / std_t

        return preprocess


def find_pretrained_weights(search_dirs: list[str] = None) -> Optional[Path]:
    """
    Search for existing pretrained weights in common locations.

    Args:
        search_dirs: List of directories to search

    Returns:
        Path to weights if found, None otherwise
    """
    if search_dirs is None:
        search_dirs = ["checkpoints", "pretrained_models", "weights", "."]

    for dir_path in search_dirs:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            continue

        for name in KNOWN_CHECKPOINT_NAMES:
            weights_path = dir_path / name
            if weights_path.exists():
                return weights_path

        # Also search for any .pt/.pth files with 'road' or 'dlinknet' in name
        for pattern in ["*road*.pt", "*road*.pth", "*dlinknet*.pt", "*dlinknet*.pth"]:
            matches = list(dir_path.glob(pattern))
            if matches:
                return matches[0]

    return None


def create_pretrained_model(
    architecture: str = "linknet",
    encoder: str = "resnet34",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    auto_find_weights: bool = True,
) -> nn.Module:
    """
    Create a road segmentation model with pretrained encoder.

    Args:
        architecture: Model architecture (unet, linknet, fpn, deeplabv3plus)
        encoder: Encoder backbone (resnet34, resnet50, efficientnet-b0, etc.)
        checkpoint_path: Optional path to fine-tuned weights
        device: Device to load model on
        auto_find_weights: If True and no checkpoint_path, search for existing weights

    Returns:
        Model ready for inference or training
    """
    model = PretrainedRoadSegmentation(
        architecture=architecture,
        encoder_name=encoder,
        encoder_weights="imagenet",
        num_classes=1,
    )

    # Auto-find weights if not specified
    if not checkpoint_path and auto_find_weights:
        found_path = find_pretrained_weights()
        if found_path:
            checkpoint_path = str(found_path)
            print(f"Auto-found pretrained weights: {found_path}")

    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            print(f"Loading checkpoint from {path}")
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)

                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Try to load with strict=False to handle architecture mismatches
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print(f"Loaded fine-tuned weights (exact match)")
                except RuntimeError:
                    # Try loading just the inner model
                    try:
                        model.model.load_state_dict(state_dict, strict=False)
                        print(f"Loaded fine-tuned weights (partial match to inner model)")
                    except RuntimeError as e:
                        print(f"Warning: Could not load weights: {e}")
                        print(f"Using ImageNet pretrained encoder only")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print(f"Using ImageNet pretrained encoder only")
    else:
        print(f"No fine-tuned weights found. Using ImageNet pretrained encoder.")
        print(f"For better results, download pretrained weights or train on DeepGlobe:")
        print(f"  1. Download: https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma")
        print(f"     Save to checkpoints/dlinknet34_deepglobe.pt")
        print(f"  2. Or train: python scripts/train_deepglobe.py --epochs 50")

    model = model.to(device).float()
    return model


def download_pretrained_weights(
    model_name: str,
    save_dir: str = "pretrained_models",
) -> Optional[Path]:
    """
    Download pretrained weights for road segmentation.

    Args:
        model_name: Name of the pretrained model
        save_dir: Directory to save weights

    Returns:
        Path to downloaded weights or None if not available
    """
    url = PRETRAINED_URLS.get(model_name)

    if url is None:
        print(f"No pretrained weights available for {model_name}")
        print("Using ImageNet pretrained encoder instead.")
        return None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    weights_path = save_path / f"{model_name}.pt"

    if weights_path.exists():
        print(f"Weights already exist at {weights_path}")
        return weights_path

    print(f"Downloading {model_name} weights from {url}...")
    try:
        urllib.request.urlretrieve(url, weights_path)
        print(f"Downloaded to {weights_path}")
        return weights_path
    except Exception as e:
        print(f"Failed to download weights: {e}")
        return None


def list_available_encoders() -> list[str]:
    """List all available encoder backbones."""
    if not SMP_AVAILABLE:
        return []

    return smp.encoders.get_encoder_names()


def get_encoder_info(encoder_name: str) -> dict:
    """Get information about a specific encoder."""
    if not SMP_AVAILABLE:
        return {}

    try:
        params = smp.encoders.get_preprocessing_params(encoder_name)
        return {
            "name": encoder_name,
            "input_space": params.get("input_space", "RGB"),
            "input_range": params.get("input_range", [0, 1]),
            "mean": params.get("mean", [0.485, 0.456, 0.406]),
            "std": params.get("std", [0.229, 0.224, 0.225]),
        }
    except Exception:
        return {"name": encoder_name, "error": "Unknown encoder"}


if __name__ == "__main__":
    # Quick test
    print("Testing pretrained model creation...")

    if SMP_AVAILABLE:
        model = create_pretrained_model("linknet", "resnet34", device="cpu")
        print(f"Created model: {model.architecture} with {model.encoder_name}")

        # Test forward pass
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = model(x)
        print(f"Input: {x.shape} -> Output: {out.shape}")

        # List some encoders
        encoders = list_available_encoders()[:10]
        print(f"\nAvailable encoders (first 10): {encoders}")
    else:
        print("segmentation_models_pytorch not available")
