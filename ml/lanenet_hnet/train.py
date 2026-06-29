#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LaneNetDataset, load_records
from losses import LaneNetLoss
from model import build_lanenet_hnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LaneNet + H-Net.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: LaneNetLoss,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    for images, binary_masks, instance_masks in tqdm(loader, leave=False):
        images = images.to(device)
        binary_masks = binary_masks.to(device)
        instance_masks = instance_masks.to(device)
        with torch.set_grad_enabled(training):
            binary_logits, embeddings, h_params = model(images)
            loss = criterion(binary_logits, embeddings, h_params, binary_masks, instance_masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    records = load_records(
        Path(dataset_cfg["split_csv"]),
        Path(dataset_cfg["image_root"]),
        Path(dataset_cfg["binary_mask_root"]),
        Path(dataset_cfg["instance_mask_root"]),
    )
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    if not train_records or not val_records:
        raise ValueError("Need at least one train and one val record in splits.csv.")

    width = int(model_cfg["input_width"])
    height = int(model_cfg["input_height"])
    train_ds = LaneNetDataset(train_records, width=width, height=height)
    val_ds = LaneNetDataset(val_records, width=width, height=height)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False)

    device = args.device
    model = build_lanenet_hnet(model_cfg).to(device)
    criterion = LaneNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    best_val = float("inf")
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_cfg": model_cfg,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pth")


if __name__ == "__main__":
    main()

