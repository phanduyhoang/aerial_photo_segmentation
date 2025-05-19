import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for multiclass segmentation.  
    Expects raw logits of shape [B, C, H, W] and target of shape [B, H, W] with class indices.
    """
    def __init__(self, weight_path: str = 'src/class_weights.pt', ignore_index: int = None):
        super().__init__()
        # Load class weights (torch tensor saved via torch.save)
        weights = torch.load(weight_path)
        self.register_buffer('weight', weights)
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: FloatTensor [B, C, H, W] - raw logits
        targets: LongTensor [B, H, W] - class indices
        """
        # Move weight to same device as inputs
        weight = self.weight.to(inputs.device)
        loss = F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        return loss

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multiclass segmentation.  
    Expects probabilities after softmax or raw logits (applies softmax internally).
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: raw logits [B, C, H, W]
        # targets: long indices [B, H, W]
        num_classes = inputs.shape[1]
        # One-hot encode targets: [B, C, H, W]
        targets_onehot = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
        # Apply softmax to logits
        probs = F.softmax(inputs, dim=1)
        # Compute per-class dice
        dims = (0,2,3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        # Dice loss is 1 - mean(dice_score)
        return 1.0 - dice_score.mean()

class CombinedLoss(nn.Module):
    """
    Combination of Weighted CrossEntropy and Dice Loss.
    """
    def __init__(self, ce_weight_path: str = 'src/class_weights.pt', alpha: float = 0.7, smooth: float = 1e-6, ignore_index: int = None):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = WeightedCrossEntropyLoss(weight_path=ce_weight_path, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets)
        dl = self.dice_loss(inputs, targets)
        return self.alpha * ce + (1 - self.alpha) * dl
