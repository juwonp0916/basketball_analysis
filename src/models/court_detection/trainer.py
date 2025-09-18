"""Training script for court detection model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import wandb

from .court_detector import CourtKeypointNet
from ..utils.losses import KeypointLoss
from ..utils.metrics import KeypointMetrics

class CourtDetectionTrainer:
    """Trainer for court detection model"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.model = CourtKeypointNet(
            num_keypoints=config['keypoints']['num_keypoints'],
            backbone=config['architecture']['backbone']
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = self._build_optimizer()

        # Initialize scheduler
        self.scheduler = self._build_scheduler()

        # Initialize loss function
        self.criterion = self._build_criterion()

        # Initialize metrics
        self.metrics = KeypointMetrics(config['keypoints']['num_keypoints'])

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        optimizer_config = self.config['training']

        if optimizer_config['optimizer'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['optimizer'] == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                momentum=0.9,
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['optimizer']}")

    def _build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler"""
        scheduler_type = self.config['training']['scheduler']

        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def _build_criterion(self) -> nn.Module:
        """Build loss function"""
        loss_config = self.config['training']['loss']

        return KeypointLoss(
            keypoint_loss_weight=loss_config['keypoint_loss_weight'],
            visibility_loss_weight=loss_config['visibility_loss_weight'],
            heatmap_loss_type=loss_config['heatmap_loss_type']
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'keypoint_accuracy': [], 'visibility_accuracy': []}

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            keypoints = batch['keypoints'].to(self.device) if 'keypoints' in batch else None
            visibility = batch['visibility'].to(self.device) if 'visibility' in batch else None

            # Forward pass
            outputs = self.model(images)

            # Calculate loss
            loss = self.criterion(outputs, {
                'keypoints': keypoints,
                'visibility': visibility
            })

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record metrics
            epoch_losses.append(loss.item())

            # Calculate accuracy metrics
            if keypoints is not None:
                metrics = self.metrics.calculate_metrics(outputs, {
                    'keypoints': keypoints,
                    'visibility': visibility
                })
                epoch_metrics['keypoint_accuracy'].append(metrics['keypoint_accuracy'])
                epoch_metrics['visibility_accuracy'].append(metrics['visibility_accuracy'])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        # Calculate epoch averages
        epoch_results = {
            'train_loss': np.mean(epoch_losses),
            'train_keypoint_accuracy': np.mean(epoch_metrics['keypoint_accuracy']) if epoch_metrics['keypoint_accuracy'] else 0,
            'train_visibility_accuracy': np.mean(epoch_metrics['visibility_accuracy']) if epoch_metrics['visibility_accuracy'] else 0
        }

        return epoch_results

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = []
        epoch_metrics = {'keypoint_accuracy': [], 'visibility_accuracy': []}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = batch['image'].to(self.device)
                keypoints = batch['keypoints'].to(self.device) if 'keypoints' in batch else None
                visibility = batch['visibility'].to(self.device) if 'visibility' in batch else None

                # Forward pass
                outputs = self.model(images)

                # Calculate loss
                loss = self.criterion(outputs, {
                    'keypoints': keypoints,
                    'visibility': visibility
                })

                epoch_losses.append(loss.item())

                # Calculate metrics
                if keypoints is not None:
                    metrics = self.metrics.calculate_metrics(outputs, {
                        'keypoints': keypoints,
                        'visibility': visibility
                    })
                    epoch_metrics['keypoint_accuracy'].append(metrics['keypoint_accuracy'])
                    epoch_metrics['visibility_accuracy'].append(metrics['visibility_accuracy'])

        # Calculate epoch averages
        epoch_results = {
            'val_loss': np.mean(epoch_losses),
            'val_keypoint_accuracy': np.mean(epoch_metrics['keypoint_accuracy']) if epoch_metrics['keypoint_accuracy'] else 0,
            'val_visibility_accuracy': np.mean(epoch_metrics['visibility_accuracy']) if epoch_metrics['visibility_accuracy'] else 0
        }

        return epoch_results

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              save_dir: str, use_wandb: bool = False) -> Dict[str, List[float]]:
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if use_wandb:
            wandb.init(project="basketball-court-detection", config=self.config)
            wandb.watch(self.model)

        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_keypoint_accuracy': [],
            'val_keypoint_accuracy': []
        }

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train epoch
            train_results = self.train_epoch(train_loader)

            # Validate epoch
            val_results = self.validate_epoch(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Record history
            training_history['train_loss'].append(train_results['train_loss'])
            training_history['val_loss'].append(val_results['val_loss'])
            training_history['train_keypoint_accuracy'].append(train_results['train_keypoint_accuracy'])
            training_history['val_keypoint_accuracy'].append(val_results['val_keypoint_accuracy'])

            # Log results
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_results['train_loss']:.4f}, "
                f"Val Loss: {val_results['val_loss']:.4f}, "
                f"Train Acc: {train_results['train_keypoint_accuracy']:.4f}, "
                f"Val Acc: {val_results['val_keypoint_accuracy']:.4f}"
            )

            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **train_results,
                    **val_results
                })

            # Save best model
            if val_results['val_loss'] < self.best_loss:
                self.best_loss = val_results['val_loss']
                self.save_checkpoint(save_dir / 'best_model.pth', epoch, is_best=True)

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch}.pth', epoch)

        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pth', self.current_epoch, is_best=False)

        if use_wandb:
            wandb.finish()

        return training_history

    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'is_best': is_best
        }

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.current_epoch = checkpoint['epoch']

        self.logger.info(f"Loaded checkpoint from {path}")

class KeypointLoss(nn.Module):
    """Loss function for keypoint detection"""

    def __init__(self, keypoint_loss_weight: float = 1.0,
                 visibility_loss_weight: float = 0.5,
                 heatmap_loss_type: str = 'mse'):
        super(KeypointLoss, self).__init__()
        self.keypoint_loss_weight = keypoint_loss_weight
        self.visibility_loss_weight = visibility_loss_weight
        self.heatmap_loss_type = heatmap_loss_type

        if heatmap_loss_type == 'mse':
            self.heatmap_loss = nn.MSELoss()
        elif heatmap_loss_type == 'focal':
            self.heatmap_loss = self._focal_loss
        else:
            raise ValueError(f"Unsupported heatmap loss type: {heatmap_loss_type}")

        self.visibility_loss = nn.BCELoss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate total loss"""
        total_loss = 0

        # Heatmap loss
        if 'keypoints' in targets and targets['keypoints'] is not None:
            pred_heatmaps = predictions['heatmaps']
            target_heatmaps = self._generate_target_heatmaps(
                targets['keypoints'], pred_heatmaps.shape
            )
            heatmap_loss = self.heatmap_loss(pred_heatmaps, target_heatmaps)
            total_loss += self.keypoint_loss_weight * heatmap_loss

        # Visibility loss
        if 'visibility' in targets and targets['visibility'] is not None:
            pred_visibility = predictions['visibility']
            target_visibility = targets['visibility']
            visibility_loss = self.visibility_loss(pred_visibility, target_visibility)
            total_loss += self.visibility_loss_weight * visibility_loss

        return total_loss

    def _generate_target_heatmaps(self, keypoints: torch.Tensor,
                                 heatmap_shape: torch.Size) -> torch.Tensor:
        """Generate target heatmaps from keypoint coordinates"""
        batch_size, num_keypoints, height, width = heatmap_shape
        device = keypoints.device

        target_heatmaps = torch.zeros(heatmap_shape, device=device)

        for b in range(batch_size):
            for k in range(num_keypoints):
                if keypoints[b, k, 2] > 0:  # Visible keypoint
                    x, y = keypoints[b, k, 0], keypoints[b, k, 1]
                    # Convert to heatmap coordinates
                    hm_x = int(x * width)
                    hm_y = int(y * height)

                    if 0 <= hm_x < width and 0 <= hm_y < height:
                        # Generate Gaussian around keypoint
                        sigma = 2
                        self._generate_gaussian(target_heatmaps[b, k], hm_x, hm_y, sigma)

        return target_heatmaps

    def _generate_gaussian(self, heatmap: torch.Tensor, center_x: int, center_y: int,
                          sigma: float) -> None:
        """Generate Gaussian heatmap around center point"""
        height, width = heatmap.shape
        x = torch.arange(width, device=heatmap.device)
        y = torch.arange(height, device=heatmap.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        # Calculate Gaussian
        gaussian = torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
        heatmap.copy_(gaussian)

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor,
                   alpha: float = 2, beta: float = 4) -> torch.Tensor:
        """Focal loss for heatmap regression"""
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        neg_weights = torch.pow(1 - target, beta)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
        neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss