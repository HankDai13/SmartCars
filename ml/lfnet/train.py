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

from dataset import LFNetDataset, load_records, split_records
from model import build_lfnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LFNet steering regressor.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []

    for images, angles in tqdm(loader, leave=False):
        images = images.to(device)
        angles = angles.to(device)
        with torch.set_grad_enabled(training):
            pred = model(images)
            loss = criterion(pred, angles)
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

    records = load_records(Path(dataset_cfg["csv"]), Path(dataset_cfg["image_root"]))
    train_records, val_records = split_records(records, float(dataset_cfg.get("val_ratio", 0.15)))

    width = int(model_cfg["input_width"])
    height = int(model_cfg["input_height"])
    train_ds = LFNetDataset(train_records, width=width, height=height)
    val_ds = LFNetDataset(val_records, width=width, height=height)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False)

    device = args.device
    model = build_lfnet(model_cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    best_val = float("inf")
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        print(f"epoch={epoch} train_mse={train_loss:.4f} val_mse={val_loss:.4f}")

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

