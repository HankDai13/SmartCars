#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_CONFIG = Path("configs/yolov5_finetune.yaml")


@dataclass
class SplitSummary:
    split: str
    images: int
    labels: int
    objects: int
    class_counts: Counter[int]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def names_from_config(config: dict[str, Any]) -> dict[int, str]:
    raw_names = config.get("dataset", {}).get("names", {})
    names: dict[int, str] = {}
    for key, value in raw_names.items():
        names[int(key)] = str(value)
    if not names:
        raise ValueError("dataset.names is empty.")
    expected = list(range(max(names) + 1))
    missing = [idx for idx in expected if idx not in names]
    if missing:
        raise ValueError(f"dataset.names has missing class ids: {missing}")
    return names


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def validate_label_file(path: Path, num_classes: int, counts: Counter[int]) -> int:
    objects = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 fields, got {len(parts)}")
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: non-numeric YOLO label field") from exc
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{path}:{line_no}: class_id {class_id} out of range")
        if any(value < 0.0 or value > 1.0 for value in coords):
            raise ValueError(f"{path}:{line_no}: bbox values must be normalized to [0, 1]")
        counts[class_id] += 1
        objects += 1
    return objects


def validate_dataset(dataset_root: Path, names: dict[int, str]) -> list[SplitSummary]:
    summaries: list[SplitSummary] = []
    num_classes = len(names)
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        if not label_dir.is_dir():
            raise FileNotFoundError(label_dir)

        images = image_files(image_dir)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        if missing_labels:
            preview = ", ".join(missing_labels[:5])
            raise ValueError(f"{split}: {len(missing_labels)} images have no label file: {preview}")
        if missing_images:
            preview = ", ".join(missing_images[:5])
            raise ValueError(f"{split}: {len(missing_images)} labels have no image file: {preview}")

        class_counts: Counter[int] = Counter()
        objects = 0
        for label_path in labels:
            objects += validate_label_file(label_path, num_classes, class_counts)
        summaries.append(
            SplitSummary(
                split=split,
                images=len(images),
                labels=len(labels),
                objects=objects,
                class_counts=class_counts,
            )
        )
    return summaries


def write_data_yaml(path: Path, dataset_root: Path, names: dict[int, str], dataset_cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(dataset_root.resolve()),
        "train": dataset_cfg.get("train", "images/train"),
        "val": dataset_cfg.get("val", "images/val"),
        "test": dataset_cfg.get("test", "images/test"),
        "names": names,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def add_optional_value(cmd: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    cmd.extend([flag, str(value)])


def add_optional_bool(cmd: list[str], flag: str, value: Any) -> None:
    if bool(value):
        cmd.append(flag)


def build_train_command(
    config: dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
    root: Path,
    data_yaml: Path,
) -> list[str]:
    yolov5_cfg = config.get("yolov5", {})
    train_cfg = dict(config.get("train", {}))

    overrides = {
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weights": args.weights,
        "device": args.device,
        "name": args.name,
    }
    for key, value in overrides.items():
        if value is not None:
            if key == "weights":
                yolov5_cfg["weights"] = value
            else:
                train_cfg[key] = value

    yolov5_dir = resolve_path(args.yolov5_dir or yolov5_cfg.get("dir", "external/yolov5"), root)
    train_py = yolov5_dir / "train.py"
    if not train_py.exists() and not args.dry_run:
        raise FileNotFoundError(
            f"{train_py} does not exist. Run: bash ml/yolov5/fetch_yolov5.sh"
        )

    project = resolve_path(train_cfg.get("project", "models/yolov5/runs"), root)
    project.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        cmd = [sys.executable, str(train_py), "--resume"]
        if args.resume is not True:
            cmd.append(str(resolve_path(args.resume, root)))
        return cmd

    cmd = [
        sys.executable,
        str(train_py),
        "--img",
        str(train_cfg.get("img_size", 640)),
        "--batch-size",
        str(train_cfg.get("batch_size", 16)),
        "--epochs",
        str(train_cfg.get("epochs", 120)),
        "--data",
        str(data_yaml),
        "--weights",
        str(yolov5_cfg.get("weights", "yolov5s.pt")),
        "--project",
        str(project),
        "--name",
        str(train_cfg.get("name", "smartcar_yolov5")),
    ]

    add_optional_value(cmd, "--workers", train_cfg.get("workers"))
    add_optional_value(cmd, "--device", train_cfg.get("device"))
    add_optional_value(cmd, "--optimizer", train_cfg.get("optimizer"))
    add_optional_value(cmd, "--patience", train_cfg.get("patience"))
    add_optional_value(cmd, "--seed", train_cfg.get("seed"))
    add_optional_value(cmd, "--save-period", train_cfg.get("save_period"))
    add_optional_value(cmd, "--freeze", train_cfg.get("freeze"))
    add_optional_value(cmd, "--label-smoothing", train_cfg.get("label_smoothing"))

    hyp = train_cfg.get("hyp")
    if hyp:
        add_optional_value(cmd, "--hyp", resolve_path(hyp, root))

    cache = train_cfg.get("cache")
    if isinstance(cache, str) and cache:
        cmd.extend(["--cache", cache])
    elif cache is True:
        cmd.append("--cache")

    add_optional_bool(cmd, "--exist-ok", train_cfg.get("exist_ok"))
    add_optional_bool(cmd, "--rect", train_cfg.get("rect"))
    add_optional_bool(cmd, "--multi-scale", train_cfg.get("multi_scale"))
    add_optional_bool(cmd, "--cos-lr", train_cfg.get("cos_lr"))

    return cmd


def print_dataset_summary(summaries: list[SplitSummary], names: dict[int, str]) -> None:
    print("Dataset summary:")
    total_counts: Counter[int] = Counter()
    for summary in summaries:
        total_counts.update(summary.class_counts)
        print(
            f"  {summary.split}: images={summary.images}, labels={summary.labels}, "
            f"objects={summary.objects}"
        )
    print("Class distribution:")
    for class_id, name in names.items():
        print(f"  {class_id}:{name}={total_counts[class_id]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate data and fine-tune YOLOv5 for SmartCars.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--yolov5-dir", type=Path)
    parser.add_argument("--weights")
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--name")
    parser.add_argument("--resume", nargs="?", const=True, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the YOLOv5 command without running it.")
    parser.add_argument("--skip-data-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    config_path = resolve_path(args.config, root)
    config = load_yaml(config_path)
    dataset_cfg = config.get("dataset", {})
    names = names_from_config(config)
    dataset_root = resolve_path(dataset_cfg.get("root", "data/yolo"), root)
    data_yaml = resolve_path(
        dataset_cfg.get("generated_data_yaml", "models/yolov5/generated/smartcar_data.yaml"),
        root,
    )

    if not args.skip_data_check:
        summaries = validate_dataset(dataset_root, names)
        print_dataset_summary(summaries, names)

    write_data_yaml(data_yaml, dataset_root, names, dataset_cfg)
    cmd = build_train_command(config, config_path, args, root, data_yaml)
    print("YOLOv5 command:")
    print(shlex.join(cmd))

    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    subprocess.run(cmd, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
