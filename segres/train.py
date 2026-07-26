import random 
import numpy as np
import torch
 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
def dice_score(preds, targets, threshold=0.5, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
 
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
 
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
 
    score = (2.0 * intersection + smooth) / (union + smooth)
    return score.mean().item()
 
 
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
 
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
 
        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item() * images.size(0)
 
    return running_loss / len(loader.dataset)
 
 
def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
 
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
 
            preds = model(images)
            loss = loss_fn(preds, masks)
 
            running_loss += loss.item() * images.size(0)
            running_dice += dice_score(preds, masks) * images.size(0)
 
    avg_loss = running_loss / len(loader.dataset)
    avg_dice = running_dice / len(loader.dataset)
    return avg_loss, avg_dice
 
 