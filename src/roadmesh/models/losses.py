"""
Loss Functions for Road Segmentation

Specialized loss functions for road extraction:
- DiceLoss: Handles class imbalance (roads are small % of image)
- BCEDiceLoss: Combines pixel-wise and region-wise losses
- ConnectivityLoss: Penalizes road network discontinuities
- CombinedLoss: Weighted combination of all losses
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Handles class imbalance better than BCE by measuring overlap
    between prediction and ground truth.

    Loss = 1 - (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (logits or probabilities), shape [B, 1, H, W]
            target: Ground truth masks, shape [B, H, W] or [B, 1, H, W]

        Returns:
            Dice loss value
        """
        # Apply sigmoid if logits
        if pred.requires_grad:
            pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate Dice coefficient
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.

    BCE provides pixel-wise supervision while Dice handles
    class imbalance at the region level.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        """
        Args:
            bce_weight: Weight for BCE component
            dice_weight: Weight for Dice component
            smooth: Smoothing for Dice calculation
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (logits), shape [B, 1, H, W]
            target: Ground truth, shape [B, H, W] or [B, 1, H, W]

        Returns:
            Combined loss value
        """
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)

        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples.

    Down-weights easy examples and focuses on hard negatives.
    Useful when road pixels are rare.

    FL(p) = -alpha * (1-p)^gamma * log(p)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (logits), shape [B, 1, H, W]
            target: Ground truth, shape [B, H, W] or [B, 1, H, W]

        Returns:
            Focal loss value
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Get probabilities
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)

        # Calculate focal weight
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()


class ConnectivityLoss(nn.Module):
    """
    Connectivity Loss for road networks.

    Penalizes discontinuities in predicted road segments.
    Uses gradient-based edge detection to identify road boundaries
    and ensures they connect properly.

    This helps maintain topological integrity of the road network.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        threshold: float = 0.5,
    ):
        """
        Args:
            kernel_size: Size of Sobel kernel for gradient computation
            threshold: Threshold for binary mask
        """
        super().__init__()
        self.threshold = threshold

        # Sobel filters for gradient computation
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # Laplacian filter for detecting endpoints/junctions
        laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('laplacian', laplacian)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (logits), shape [B, 1, H, W]
            target: Ground truth, shape [B, H, W] or [B, 1, H, W]

        Returns:
            Connectivity loss value
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Get probabilities
        pred_prob = torch.sigmoid(pred)

        # Compute gradients for prediction
        pred_grad_x = F.conv2d(pred_prob, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_prob, self.sobel_y, padding=1)
        pred_grad_mag = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)

        # Compute gradients for target
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        target_grad_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)

        # Edge alignment loss (edges should be in same places)
        edge_loss = F.mse_loss(pred_grad_mag, target_grad_mag)

        # Direction consistency loss (gradients should point same direction)
        # Normalize gradients
        pred_dir = torch.cat([pred_grad_x, pred_grad_y], dim=1)
        target_dir = torch.cat([target_grad_x, target_grad_y], dim=1)

        pred_norm = pred_dir / (torch.norm(pred_dir, dim=1, keepdim=True) + 1e-8)
        target_norm = target_dir / (torch.norm(target_dir, dim=1, keepdim=True) + 1e-8)

        # Mask to only consider areas where target has edges
        edge_mask = (target_grad_mag > 0.1).float()

        # Direction loss (1 - cosine similarity)
        direction_sim = (pred_norm * target_norm).sum(dim=1, keepdim=True)
        direction_loss = ((1 - direction_sim) * edge_mask).sum() / (edge_mask.sum() + 1e-8)

        # Combine losses
        total_loss = edge_loss + 0.5 * direction_loss

        return total_loss


class CombinedLoss(nn.Module):
    """
    Combined Loss for road segmentation training.

    Combines:
    - BCE: Pixel-wise classification
    - Dice: Region-level overlap (handles class imbalance)
    - Connectivity: Topological consistency of road network

    Default weights are tuned for road extraction.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.3,
        connectivity_weight: float = 0.2,
        smooth: float = 1.0,
    ):
        """
        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            connectivity_weight: Weight for Connectivity loss
            smooth: Smoothing for Dice calculation
        """
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.connectivity_weight = connectivity_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.connectivity = ConnectivityLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred: Predictions (logits), shape [B, 1, H, W]
            target: Ground truth, shape [B, H, W] or [B, 1, H, W]

        Returns:
            Dictionary with total loss and component breakdown
        """
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Calculate individual losses
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        conn_loss = self.connectivity(pred, target)

        # Weighted combination
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.connectivity_weight * conn_loss
        )

        return {
            'total': total_loss,
            'bce': bce_loss,
            'dice': dice_loss,
            'connectivity': conn_loss,
        }


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss.

    Allows controlling the balance between false positives
    and false negatives, useful for very imbalanced data.

    TL = 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1.0,
    ):
        """
        Args:
            alpha: Weight for false positives (higher = penalize FP more)
            beta: Weight for false negatives (higher = penalize FN more)
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions (logits), shape [B, 1, H, W]
            target: Ground truth, shape [B, H, W] or [B, 1, H, W]

        Returns:
            Tversky loss value
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)

        pred_prob = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        # Calculate TP, FP, FN
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        # Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1 - tversky


if __name__ == "__main__":
    # Quick test
    print("Testing loss functions...")

    batch_size = 2
    pred = torch.randn(batch_size, 1, 512, 512)
    target = (torch.rand(batch_size, 512, 512) > 0.8).float()  # Sparse roads

    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Road coverage: {target.mean():.1%}")

    # Test each loss
    dice_loss = DiceLoss()
    print(f"Dice Loss: {dice_loss(pred, target):.4f}")

    bce_dice_loss = BCEDiceLoss()
    print(f"BCE+Dice Loss: {bce_dice_loss(pred, target):.4f}")

    focal_loss = FocalLoss()
    print(f"Focal Loss: {focal_loss(pred, target):.4f}")

    conn_loss = ConnectivityLoss()
    print(f"Connectivity Loss: {conn_loss(pred, target):.4f}")

    combined_loss = CombinedLoss()
    losses = combined_loss(pred, target)
    print(f"Combined Loss: {losses['total']:.4f}")
    print(f"  - BCE: {losses['bce']:.4f}")
    print(f"  - Dice: {losses['dice']:.4f}")
    print(f"  - Connectivity: {losses['connectivity']:.4f}")
