"""
RoadMesh Neural Network Architectures

D-LinkNet34 implementation for road segmentation.
Based on: https://github.com/zlkanata/DeepGlobe-Road-Extraction-Challenge
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Dblock(nn.Module):
    """Dilated convolution block with multiple dilation rates."""

    def __init__(self, channel: int):
        super().__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out), inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out), inplace=True)
        dilate4_out = F.relu(self.dilate4(dilate3_out), inplace=True)
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DLinkNet34(nn.Module):
    """
    D-LinkNet with ResNet34 backbone for road segmentation.

    Paper: D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution
           for High Resolution Satellite Imagery Road Extraction

    Args:
        num_classes: Number of output classes (1 for binary segmentation)
        pretrained: Whether to use pretrained ResNet34 backbone
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet34
        try:
            if pretrained:
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                resnet = models.resnet34(weights=weights)
            else:
                resnet = models.resnet34(weights=None)
        except Exception:
            # Fallback for older torchvision versions
            resnet = models.resnet34(pretrained=pretrained)

        # Encoder (ResNet34 layers)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels

        # Center block with dilated convolutions
        self.dblock = Dblock(512)

        # Decoder blocks
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        # Final layers
        self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder with skip connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final layers
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str = "cpu",
    strict: bool = False,
) -> nn.Module:
    """
    Load checkpoint weights into model.

    Handles different checkpoint formats:
    - Direct state_dict
    - Dict with 'state_dict' key
    - Dict with 'model_state_dict' key

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load weights to
        strict: Whether to strictly match keys

    Returns:
        Model with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[MODEL] Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state_dict from different formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Assume the dict is the state_dict itself
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean up keys (remove 'module.' prefix from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load weights
    try:
        missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
        if missing:
            print(f"[MODEL] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[MODEL] Unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"[MODEL] Warning: Could not load all weights: {e}")
        # Try to load what we can
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"[MODEL] Loaded {len(pretrained_dict)}/{len(model_dict)} layers")

    print(f"[MODEL] Checkpoint loaded successfully")
    return model


def create_model(
    architecture: str = "dlinknet34",
    num_classes: int = 1,
    pretrained: bool = True,
    checkpoint_path: Optional[str | Path] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Create and optionally load pretrained model.

    Args:
        architecture: Model architecture ('dlinknet34', 'unet_resnet34')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained backbone
        checkpoint_path: Optional path to trained weights
        device: Device to put model on

    Returns:
        Initialized model
    """
    print(f"[MODEL] Creating {architecture} model...")

    if architecture.lower() == "dlinknet34":
        model = DLinkNet34(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Load checkpoint if provided
    if checkpoint_path:
        model = load_checkpoint(model, checkpoint_path, device=device)

    # Move to device
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total parameters: {total_params:,}")
    print(f"[MODEL] Trainable parameters: {trainable_params:,}")

    return model
