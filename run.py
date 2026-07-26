import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from segres import (
    SolarFilamentDataset,
    FileGroupedSampler,
    build_model,
    get_train_transform,
    get_val_transform,
    DiceBCELoss,
    set_seed,
    train_one_epoch,
    validate,
    load_model,
    predict_tiles,
    build_submission,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def list_image_files(images_dir):
    return sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpeg", ".jpg", ".png")))


def build_loaders(cfg):
    file_names = list_image_files(cfg["train_images_dir"])
    train_files, val_files = train_test_split(
        file_names, test_size=cfg["val_fraction"], random_state=cfg["seed"]
    )

    common_kwargs = dict(
        images_dir=cfg["train_images_dir"],
        annotations_json=cfg["train_annotations"],
        tile_size=cfg["tile_size"],
        overlap=cfg["overlap"],
        image_height=cfg["image_height"],
        image_width=cfg["image_width"],
    )

    train_dataset = SolarFilamentDataset(
        train_files, transform=get_train_transform(), **common_kwargs
    )
    val_dataset = SolarFilamentDataset(
        val_files, transform=get_val_transform(), return_meta=True, **common_kwargs
    )

    # shuffle=True at the DataLoader level shuffles individual tiles, defeating
    # the dataset's per-file image/mask cache (see segres/dataset.py). Shuffle
    # file order instead, keeping a file's tiles clustered together.
    train_sampler = FileGroupedSampler(train_dataset, seed=cfg["seed"])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], sampler=train_sampler,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


def train(cfg, device, resume=False):
    train_loader, val_loader, train_sampler = build_loaders(cfg)

    model = build_model().to(device)
    if resume:
        if os.path.exists(cfg["checkpoint_path"]):
            model.load_state_dict(torch.load(cfg["checkpoint_path"], map_location=device))
            print(f"resumed weights from {cfg['checkpoint_path']}")
        else:
            print(f"--resume given but no checkpoint at {cfg['checkpoint_path']}, starting fresh")

    loss_fn = DiceBCELoss(pos_weight=cfg["pos_weight"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # maximize val dice; halve LR after `lr_patience` epochs without improvement
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=cfg["lr_patience"]
    )
    amp_enabled = cfg["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)
    best_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler=scaler)
        val_loss, metrics = validate(
            model, val_loader, loss_fn, device,
            threshold=cfg["pred_threshold"], min_area=cfg["min_area"],
        )
        scheduler.step(metrics["dice"])

        print(
            f"epoch {epoch}/{cfg['epochs']} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"dice={metrics['dice']:.4f} (n={metrics['n_nonempty_images']}) "
            f"empty_correct={metrics['empty_correct_frac']:.4f} (n={metrics['n_empty_images']}) "
            f"instance_f1={metrics['instance_f1']:.4f} "
            f"(precision={metrics['instance_precision']:.4f} recall={metrics['instance_recall']:.4f})"
        )

        # checkpoint on full-image dice over images that actually contain a
        # filament, not per-tile dice inflated by empty-tile smoothing
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), cfg["checkpoint_path"])
            print(f"  saved new best checkpoint (dice={best_dice:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg["early_stop_patience"]:
                print(f"  no improvement in {epochs_without_improvement} epochs, stopping early")
                break

    return best_dice


def predict(cfg, device):
    os.makedirs(os.path.dirname(cfg["submission_path"]), exist_ok=True)

    test_files = list_image_files(cfg["test_images_dir"])
    test_dataset = SolarFilamentDataset(
        test_files,
        images_dir=cfg["test_images_dir"],
        transform=get_val_transform(),
        is_test=True,
        tile_size=cfg["tile_size"],
        overlap=cfg["overlap"],
        image_height=cfg["image_height"],
        image_width=cfg["image_width"],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    model = load_model(cfg["checkpoint_path"], device)
    results = predict_tiles(model, test_loader, device)
    build_submission(
        results,
        threshold=cfg["pred_threshold"],
        min_area=cfg["min_area"],
        output_csv=cfg["submission_path"],
        tile_size=cfg["tile_size"],
        image_height=cfg["image_height"],
        image_width=cfg["image_width"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train and/or run inference for MAGFiLO filament segmentation.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", help="Only train, skip prediction.")
    mode.add_argument("--predict-only", action="store_true", help="Only run inference from the existing checkpoint, skip training.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the existing checkpoint instead of from scratch.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    if not args.predict_only:
        train(cfg, device, resume=args.resume)
    if not args.train_only:
        predict(cfg, device)


if __name__ == "__main__":
    main()
