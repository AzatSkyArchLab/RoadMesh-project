"""
Training Pipeline

Optimized for 8GB VRAM with:
- Mixed precision training (FP16)
- Gradient accumulation
- Efficient data loading
- Checkpoint management
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from roadmesh.core.config import TrainingConfig, RoadMeshConfig
from roadmesh.models.architectures import create_model
from roadmesh.models.losses import CombinedLoss
from roadmesh.data.dataset import RoadDataset, create_dataloaders


class Trainer:
    """
    Training loop with all optimizations for 8GB VRAM.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
        experiment_name: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        
        # Loss function
        self.criterion = CombinedLoss(
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight,
            connectivity_weight=config.connectivity_weight,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Checkpoint directory
        self.checkpoint_dir = config.checkpoint_dir / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'lr': [],
        }
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
            )
        elif self.config.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.config.scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
            )
        else:
            return None
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {'bce': 0.0, 'dice': 0.0, 'connectivity': 0.0}
        
        accumulation_steps = self.config.gradient_accumulation
        self.optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Mixed precision forward pass
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    losses = self.criterion(outputs, masks)
                    loss = losses['total'] / accumulation_steps
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, masks)
                loss = losses['total'] / accumulation_steps
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Accumulate metrics
            total_loss += losses['total'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average losses
        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return {'loss': avg_loss, **avg_components}
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        total_iou = 0.0
        total_precision = 0.0
        total_recall = 0.0
        
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(images)
            else:
                outputs = self.model(images)
            
            losses = self.criterion(outputs, masks)
            total_loss += losses['total'].item()
            
            # Compute IoU
            preds = (torch.sigmoid(outputs) > 0.5).float()
            iou = self._compute_iou(preds, masks)
            total_iou += iou
            
            # Compute precision/recall
            precision, recall = self._compute_precision_recall(preds, masks)
            total_precision += precision
            total_recall += recall
        
        n_batches = len(dataloader)
        
        return {
            'loss': total_loss / n_batches,
            'iou': total_iou / n_batches,
            'precision': total_precision / n_batches,
            'recall': total_recall / n_batches,
        }
    
    def _compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Intersection over Union."""
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + 1e-8) / (union + 1e-8)
        return iou.item()
    
    def _compute_precision_recall(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple[float, float]:
        """Compute precision and recall."""
        pred = pred.view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        precision = (tp + 1e-8) / (tp + fp + 1e-8)
        recall = (tp + 1e-8) / (tp + fn + 1e-8)
        
        return precision.item(), recall.item()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'history': self.history,
            'config': self.config.model_dump() if hasattr(self.config, 'model_dump') else {},
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.history = checkpoint.get('history', self.history)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (uses config if not specified)
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.epochs
        
        print(f"\n{'='*60}")
        print(f"Starting training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['iou'])
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            
            # Check for improvement
            is_best = val_metrics['iou'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['iou']
                self.epochs_without_improvement = 0
                print(f"  ✓ New best IoU: {self.best_metric:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch:03d}.pt", is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        print(f"\nTraining complete. Best IoU: {self.best_metric:.4f}")
        
        return self.history


def train_from_config(config_path: Path) -> dict:
    """
    Train model from configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Training history
    """
    # Load config
    config = RoadMeshConfig.from_yaml(config_path)
    
    # Create model
    model = create_model(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained_backbone,
    )
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        config.data.processed_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Train
    history = trainer.fit(train_loader, val_loader)
    
    return history
