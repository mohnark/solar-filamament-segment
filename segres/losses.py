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

        The .float() casts are load-bearing under AMP: inside torch.autocast
        these tensors arrive as fp16, and summing a 512x512 tile (262144
        pixels) overflows fp16's 65504 max as soon as the mean sigmoid output
        exceeds ~0.25. The overflowed union pins dice_loss at 1.0 and sends
        NaN gradients back, which GradScaler then skips — every step, forever,
        with no error raised.
        """
        preds = torch.sigmoid(preds.float())
        targets = targets.float()

        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0, pos_weight=None):
        """
        pos_weight: weight on the positive-class BCE term, to counter
        filament pixels being ~1:546 rare against background. Without it,
        plain BCE at 0.5 pushes the model toward predicting all-background.
        Registered as a buffer so it moves with the loss module's `.to(device)`.
        """
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight = None
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, preds, targets):
        d_loss = self.dice_loss(preds, targets)
        b_loss = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=self.pos_weight)
        return self.dice_weight * d_loss + self.bce_weight * b_loss
