import random
import numpy as np
import torch

from .tiling import stitch_predictions
from .predict import label_and_filter


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None):
    """
    scaler: a torch.amp.GradScaler. Pass one with enabled=True (only
    meaningful on CUDA) to train under autocast + mixed precision; pass one
    with enabled=False (or None) to train in plain fp32.
    """
    model.train()
    running_loss = 0.0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            preds = model(images)
            loss = loss_fn(preds, masks)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def _instance_match_counts(pred_instances, gt_instances, iou_thresh):
    """Greedy IoU matching between predicted and ground-truth instances.

    Returns (tp, fp, fn) counts for this one image.
    """
    matched_gt = set()
    tp = 0
    for p_inst in pred_instances:
        best_iou, best_j = 0.0, -1
        for j, g_inst in enumerate(gt_instances):
            if j in matched_gt:
                continue
            intersection = np.logical_and(p_inst, g_inst).sum()
            union = np.logical_or(p_inst, g_inst).sum()
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)

    fp = len(pred_instances) - tp
    fn = len(gt_instances) - len(matched_gt)
    return tp, fp, fn


def validate(model, loader, loss_fn, device, threshold=0.5, min_area=20, iou_thresh=0.5):
    """
    Runs validation loss per-tile (cheap, matches training objective), but
    computes dice and instance metrics on full stitched images, matching how
    the competition actually scores predictions.

    `loader` must be built with SolarFilamentDataset(..., return_meta=True)
    so tiles carry (file_name, y, x) for stitching.

    Per-tile dice with additive smoothing scores an empty tile 1.0 regardless
    of the prediction (72.8% of tiles have no filament), so an all-background
    model floors near 0.73 and checkpoint selection rewards collapsing to
    background. Stitching first and reporting empty/non-empty images
    separately avoids that: dice is only computed where there's a filament to
    find, and an all-background model scores 0 there instead of ~0.73.
    """
    dataset = loader.dataset
    model.eval()
    running_loss = 0.0

    results = {}  # file_name -> {"pred_tiles": [...], "gt_tiles": [...], "coords": [...]}

    with torch.no_grad():
        for images, masks, file_names, ys, xs in loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(preds).cpu().numpy()[:, 0]  # (B, H, W)
            gt = masks.cpu().numpy()[:, 0]  # (B, H, W)

            for i, file_name in enumerate(file_names):
                entry = results.setdefault(file_name, {"pred_tiles": [], "gt_tiles": [], "coords": []})
                entry["pred_tiles"].append(probs[i])
                entry["gt_tiles"].append(gt[i])
                entry["coords"].append((ys[i].item(), xs[i].item()))

    avg_loss = running_loss / len(loader.dataset)

    dices = []  # only images with at least one ground-truth filament pixel
    empty_total = 0
    empty_correct = 0
    tp = fp = fn = 0

    for entry in results.values():
        stitched_pred = stitch_predictions(
            entry["pred_tiles"], entry["coords"], dataset.image_height, dataset.image_width, dataset.tile_size
        )
        stitched_gt = stitch_predictions(
            entry["gt_tiles"], entry["coords"], dataset.image_height, dataset.image_width, dataset.tile_size
        )

        pred_binary, pred_instances = label_and_filter((stitched_pred > threshold).astype(np.uint8), min_area=min_area)
        gt_binary, gt_instances = label_and_filter((stitched_gt > 0.5).astype(np.uint8), min_area=1)

        if gt_binary.sum() == 0:
            empty_total += 1
            if pred_binary.sum() == 0:
                empty_correct += 1
        else:
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = pred_binary.sum() + gt_binary.sum()
            dices.append(2.0 * intersection / union if union > 0 else 1.0)

        img_tp, img_fp, img_fn = _instance_match_counts(pred_instances, gt_instances, iou_thresh)
        tp += img_tp
        fp += img_fp
        fn += img_fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    instance_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "n_nonempty_images": len(dices),
        "n_empty_images": empty_total,
        "empty_correct_frac": empty_correct / empty_total if empty_total > 0 else 1.0,
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_f1": instance_f1,
    }
    return avg_loss, metrics

