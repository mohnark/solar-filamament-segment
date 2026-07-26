import json
import os

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from segres import (
    SolarFilamentDataset,
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
        val_files, transform=get_val_transform(), **common_kwargs
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


def train(cfg, device):
    train_loader, val_loader = build_loaders(cfg)

    model = build_model().to(device)
    loss_fn = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    os.makedirs(os.path.dirname(cfg["checkpoint_path"]), exist_ok=True)
    best_dice = -1.0

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)

        print(
            f"epoch {epoch}/{cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), cfg["checkpoint_path"])
            print(f"  saved new best checkpoint (val_dice={best_dice:.4f})")

    return best_dice


def predict(cfg, device):
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
    )


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train(cfg, device)
    predict(cfg, device)


if __name__ == "__main__":
    main()
