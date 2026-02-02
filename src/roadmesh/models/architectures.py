"""
RoadMesh Neural Network Architectures

Multiple architectures for road segmentation:
- D-LinkNet34 (DeepGlobe winner)
- UNet with various backbones (via segmentation_models_pytorch)
- FPN, DeepLabV3+ and more
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Try to import segmentation_models_pytorch for additional architectures
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("[MODEL] segmentation_models_pytorch not installed. Run: pip install segmentation-models-pytorch")


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
    print(f"[MODEL] File size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state_dict from different formats
    if isinstance(checkpoint, dict):
        print(f"[MODEL] Checkpoint keys: {list(checkpoint.keys())[:10]}...")
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

    print(f"[MODEL] State dict has {len(state_dict)} keys")

    # Show first few keys from checkpoint
    ckpt_keys = list(state_dict.keys())[:5]
    print(f"[MODEL] Checkpoint first keys: {ckpt_keys}")

    # Show first few keys from model
    model_keys = list(model.state_dict().keys())[:5]
    print(f"[MODEL] Model first keys: {model_keys}")

    # Clean up keys (remove 'module.' prefix from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Check key overlap
    model_dict = model.state_dict()
    matched_keys = set(new_state_dict.keys()) & set(model_dict.keys())
    print(f"[MODEL] Matched keys: {len(matched_keys)}/{len(model_dict)}")

    # Load weights
    try:
        missing, unexpected = model.load_state_dict(new_state_dict, strict=strict)
        if missing:
            print(f"[MODEL] Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"[MODEL] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        if not missing and not unexpected:
            print(f"[MODEL] All keys matched perfectly!")
    except Exception as e:
        print(f"[MODEL] Warning: Could not load all weights: {e}")
        # Try to load what we can
        pretrained_dict = {k: v for k, v in new_state_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"[MODEL] Loaded {len(pretrained_dict)}/{len(model_dict)} layers")

    print(f"[MODEL] Checkpoint loaded successfully")
    return model


# Registry of available architectures
ARCHITECTURE_REGISTRY = {
    "dlinknet34": "D-LinkNet with ResNet34 (DeepGlobe winner)",
    "unet_resnet34": "UNet with ResNet34 encoder",
    "unet_resnet50": "UNet with ResNet50 encoder",
    "unet_efficientnet-b3": "UNet with EfficientNet-B3 encoder",
    "unet_efficientnet-b4": "UNet with EfficientNet-B4 encoder",
    "fpn_resnet34": "FPN with ResNet34 encoder",
    "fpn_resnet50": "FPN with ResNet50 encoder",
    "deeplabv3plus_resnet50": "DeepLabV3+ with ResNet50 encoder",
    "deeplabv3plus_efficientnet-b4": "DeepLabV3+ with EfficientNet-B4 encoder",
}


def list_architectures() -> dict[str, str]:
    """List all available architectures."""
    available = {"dlinknet34": ARCHITECTURE_REGISTRY["dlinknet34"]}

    if SMP_AVAILABLE:
        for name, desc in ARCHITECTURE_REGISTRY.items():
            if name != "dlinknet34":
                available[name] = desc

    return available


def create_model(
    architecture: str = "dlinknet34",
    num_classes: int = 1,
    pretrained: bool = True,
    checkpoint_path: Optional[str | Path] = None,
    device: str = "cpu",
    encoder_weights: str = "imagenet",
) -> nn.Module:
    """
    Create and optionally load pretrained model.

    Args:
        architecture: Model architecture. Options:
            - 'dlinknet34': D-LinkNet with ResNet34 (DeepGlobe)
            - 'unet_resnet34': UNet with ResNet34
            - 'unet_resnet50': UNet with ResNet50
            - 'unet_efficientnet-b3': UNet with EfficientNet-B3
            - 'fpn_resnet34': FPN with ResNet34
            - 'deeplabv3plus_resnet50': DeepLabV3+ with ResNet50
        num_classes: Number of output classes
        pretrained: Whether to use pretrained backbone
        checkpoint_path: Optional path to trained weights
        device: Device to put model on
        encoder_weights: Encoder pretrained weights ('imagenet' or None)

    Returns:
        Initialized model
    """
    print(f"[MODEL] Creating {architecture} model...")
    arch_lower = architecture.lower()

    if arch_lower == "dlinknet34":
        model = DLinkNet34(num_classes=num_classes, pretrained=pretrained)

    elif arch_lower.startswith("unet_"):
        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch required. Run: pip install segmentation-models-pytorch")
        encoder = arch_lower.replace("unet_", "")
        weights = encoder_weights if pretrained else None
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
        print(f"[MODEL] Using UNet with {encoder} encoder")

    elif arch_lower.startswith("fpn_"):
        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch required. Run: pip install segmentation-models-pytorch")
        encoder = arch_lower.replace("fpn_", "")
        weights = encoder_weights if pretrained else None
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
        print(f"[MODEL] Using FPN with {encoder} encoder")

    elif arch_lower.startswith("deeplabv3plus_"):
        if not SMP_AVAILABLE:
            raise ImportError("segmentation_models_pytorch required. Run: pip install segmentation-models-pytorch")
        encoder = arch_lower.replace("deeplabv3plus_", "")
        weights = encoder_weights if pretrained else None
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
        print(f"[MODEL] Using DeepLabV3+ with {encoder} encoder")

    else:
        available = list(list_architectures().keys())
        raise ValueError(f"Unknown architecture: {architecture}. Available: {available}")

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
