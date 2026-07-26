import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
 
    def forward(self, preds, targets):
        """
        preds: raw model output (logits), shape (B, 1, H, W)
        targets: ground-truth mask, values 0 or 1, shape (B, 1, H, W)
        """
        preds = torch.sigmoid(preds)
 
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
 
        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
 
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score
 
        return dice_loss.mean()
 
 
class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
 
    def forward(self, preds, targets):
        d_loss = self.dice_loss(preds, targets)
        b_loss = self.bce_loss(preds, targets)
        return self.dice_weight * d_loss + self.bce_weight * b_loss
