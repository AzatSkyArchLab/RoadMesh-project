"""
Neural Network Architectures for Road Segmentation

Implements D-LinkNet and UNet variants optimized for road extraction.
D-LinkNet won the DeepGlobe Road Extraction Challenge.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights


class DilatedBlock(nn.Module):
    """
    Dilated convolution block used in D-LinkNet center.

    Uses multiple dilation rates to capture multi-scale context
    without losing resolution.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        # Dilated convolutions with different rates
        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                  dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                  dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                  dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                  dilation=8, padding=8)

        # Batch normalization for each
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn4 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cascaded dilated convolutions
        d1 = self.relu(self.bn1(self.dilate1(x)))
        d2 = self.relu(self.bn2(self.dilate2(d1)))
        d3 = self.relu(self.bn3(self.dilate3(d2)))
        d4 = self.relu(self.bn4(self.dilate4(d3)))

        # Sum all dilated outputs (residual-like connection)
        out = x + d1 + d2 + d3 + d4
        return out


class DecoderBlock(nn.Module):
    """
    Decoder block with transposed convolution for upsampling.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        # Upsample
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        # Merge with skip connection and refine
        merged_channels = in_channels // 2 + skip_channels
        self.conv1 = nn.Conv2d(merged_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Handle size mismatch
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        # Refine
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x


class DLinkNet34(nn.Module):
    """
    D-LinkNet with ResNet34 encoder.

    Architecture designed specifically for road extraction:
    - ResNet34 encoder (pretrained on ImageNet)
    - Dilated convolution center for multi-scale context
    - Decoder with skip connections

    Reference: "D-LinkNet: LinkNet with Pretrained Encoder and
    Dilated Convolution for High Resolution Satellite Imagery Road Extraction"
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Load pretrained ResNet34
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
            resnet = models.resnet34(weights=weights)
        else:
            resnet = models.resnet34(weights=None)

        # Encoder (ResNet34 layers)
        self.encoder0 = nn.Sequential(
            resnet.conv1,   # 64, H/2
            resnet.bn1,
            resnet.relu,
        )
        self.pool0 = resnet.maxpool  # H/4

        self.encoder1 = resnet.layer1  # 64,  H/4
        self.encoder2 = resnet.layer2  # 128, H/8
        self.encoder3 = resnet.layer3  # 256, H/16
        self.encoder4 = resnet.layer4  # 512, H/32

        # Center with dilated convolutions
        self.center = DilatedBlock(512)

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        # Final layers
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # Initialize decoder weights
        self._init_decoder()

    def _init_decoder(self):
        """Initialize decoder weights using Kaiming initialization."""
        for module in [self.center, self.decoder4, self.decoder3,
                       self.decoder2, self.decoder1, self.final_upsample,
                       self.final_conv]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.encoder0(x)      # H/2
        e0_pool = self.pool0(e0)   # H/4

        e1 = self.encoder1(e0_pool)  # H/4
        e2 = self.encoder2(e1)       # H/8
        e3 = self.encoder3(e2)       # H/16
        e4 = self.encoder4(e3)       # H/32

        # Center with dilated convolutions
        center = self.center(e4)

        # Decoder with skip connections
        d4 = self.decoder4(center, e3)  # H/16
        d3 = self.decoder3(d4, e2)       # H/8
        d2 = self.decoder2(d3, e1)       # H/4
        d1 = self.decoder1(d2, e0)       # H/2

        # Final
        out = self.final_upsample(d1)    # H
        out = self.dropout(out)
        out = self.final_conv(out)

        return out


class UNetResNet34(nn.Module):
    """
    Standard U-Net with ResNet34 encoder.

    Alternative to D-LinkNet, simpler but still effective.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Load pretrained ResNet34
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
            resnet = models.resnet34(weights=weights)
        else:
            resnet = models.resnet34(weights=None)

        # Encoder
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.pool0 = resnet.maxpool

        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # Center
        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        # Final
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.encoder0(x)
        e0_pool = self.pool0(e0)

        e1 = self.encoder1(e0_pool)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        center = self.center(e4)

        # Decoder
        d4 = self.decoder4(center, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        # Final
        out = self.final_upsample(d1)
        out = self.dropout(out)
        out = self.final_conv(out)

        return out


def create_model(
    architecture: Literal["dlinknet34", "unet_resnet34"] = "dlinknet34",
    num_classes: int = 1,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to create model.

    Args:
        architecture: Model architecture ("dlinknet34" or "unet_resnet34")
        num_classes: Number of output classes (1 for binary segmentation)
        pretrained: Use ImageNet pretrained weights for encoder
        checkpoint_path: Path to load trained weights from
        device: Device to place model on ("cpu" or "cuda")
        dropout: Dropout probability

    Returns:
        Initialized model on specified device
    """
    # Create model
    if architecture == "dlinknet34":
        model = DLinkNet34(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    elif architecture == "unet_resnet34":
        model = UNetResNet34(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Choose from: dlinknet34, unet_resnet34")

    # Load checkpoint if provided
    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print(f"Loaded checkpoint from {path}")
        else:
            print(f"Warning: Checkpoint not found at {path}")

    # Move to device and ensure FP32 (checkpoint may have FP16 from mixed precision training)
    model = model.to(device).float()

    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_mb": total * 4 / (1024 * 1024),  # Assuming float32
    }


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int = 4,
    input_size: int = 512,
    mixed_precision: bool = True,
) -> dict:
    """
    Estimate GPU memory usage for training.

    Args:
        model: PyTorch model
        batch_size: Batch size
        input_size: Input image size (assumes square)
        mixed_precision: Whether using FP16 mixed precision

    Returns:
        Dictionary with memory estimates in MB
    """
    params = count_parameters(model)

    # Model parameters
    param_memory = params["total_mb"]
    if mixed_precision:
        param_memory *= 0.75  # FP16 for forward, FP32 for master weights

    # Optimizer states (Adam has 2 states per parameter)
    optimizer_memory = param_memory * 2

    # Gradients
    gradient_memory = param_memory

    # Activations (rough estimate based on typical segmentation model)
    # This is highly architecture dependent
    bytes_per_element = 2 if mixed_precision else 4
    activation_memory = (
        batch_size * input_size * input_size * 512 * bytes_per_element / (1024 * 1024)
    )

    total = param_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "parameters_mb": param_memory,
        "optimizer_mb": optimizer_memory,
        "gradients_mb": gradient_memory,
        "activations_mb": activation_memory,
        "total_estimated_mb": total,
        "recommended_batch_size_8gb": max(1, int(8000 / (total / batch_size))),
    }


if __name__ == "__main__":
    # Quick test
    print("Testing D-LinkNet34...")
    model = create_model("dlinknet34", pretrained=False, device="cpu")

    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    params = count_parameters(model)
    print(f"Parameters: {params['total']:,} ({params['total_mb']:.1f} MB)")

    memory = estimate_memory_usage(model, batch_size=4)
    print(f"Estimated memory: {memory['total_estimated_mb']:.0f} MB")
    print(f"Recommended batch size for 8GB: {memory['recommended_batch_size_8gb']}")
