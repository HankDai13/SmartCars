#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path


DEFAULT_RUNS = Path("models/yolov5/runs")
ARTIFACTS = [
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "PR_curve.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "labels.jpg",
    "train_batch0.jpg",
    "val_batch0_pred.jpg",
]
METRIC_KEYS = [
    "epoch",
    "metrics/precision",
    "metrics/recall",
    "metrics/mAP_0.5",
    "metrics/mAP_0.5:0.95",
    "val/box_loss",
    "val/obj_loss",
    "val/cls_loss",
    "train/box_loss",
    "train/obj_loss",
    "train/cls_loss",
]


def latest_run(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No YOLOv5 run directories found under {runs_dir}")
    return max(candidates, key=lambda p: (p / "results.csv").stat().st_mtime if (p / "results.csv").exists() else p.stat().st_mtime)


def read_last_metrics(run_dir: Path) -> dict[str, str]:
    results = run_dir / "results.csv"
    if not results.exists():
        return {}
    last: dict[str, str] = {}
    with results.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = {key.strip(): value.strip() for key, value in row.items() if key is not None}
    return last


def print_status(run_dir: Path, runs_dir: Path) -> None:
    metrics = read_last_metrics(run_dir)
    print(f"Run: {run_dir}")
    if metrics:
        print("Latest metrics:")
        for key in METRIC_KEYS:
            if key in metrics:
                print(f"  {key}: {metrics[key]}")
    else:
        print("Latest metrics: results.csv not written yet")

    print("Artifacts:")
    for rel_path in ARTIFACTS:
        path = run_dir / rel_path
        marker = "ok" if path.exists() else "--"
        print(f"  [{marker}] {path}")

    print("TensorBoard:")
    print(f"  tensorboard --logdir {runs_dir} --host 0.0.0.0 --port 6006")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show YOLOv5 training progress and artifact locations.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--run", type=Path, help="Specific run directory. Defaults to latest run.")
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh interval in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        try:
            run_dir = args.run or latest_run(args.runs_dir)
        except FileNotFoundError as exc:
            print(exc)
            print("Start a training run first, for example:")
            print("  python ml/yolov5/train_smartcar.py --epochs 5 --batch-size 8 --name smoke_yolov5s")
            raise SystemExit(1) from exc
        if args.watch and os.isatty(1):
            print("\033[2J\033[H", end="")
        print_status(run_dir, args.runs_dir)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
