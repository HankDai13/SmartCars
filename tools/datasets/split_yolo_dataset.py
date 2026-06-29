#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO images and labels into train/val/test.")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train", default=0.8, type=float)
    parser.add_argument("--val", default=0.1, type=float)
    parser.add_argument("--test", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving/symlinking.")
    return parser.parse_args()


def ensure_ratio(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def link_or_copy(src: Path, dst: Path, copy_file: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_file:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    ensure_ratio(args.train, args.val, args.test)

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in {args.images}")

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []
    for image in images:
        label = args.labels / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
        else:
            missing_labels.append(image)

    if missing_labels:
        preview = ", ".join(p.name for p in missing_labels[:5])
        raise FileNotFoundError(f"{len(missing_labels)} images have no YOLO label. First: {preview}")

    random.seed(args.seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * args.train)
    n_val = int(n_total * args.val)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split, split_pairs in splits.items():
        for image, label in split_pairs:
            link_or_copy(image, args.output / "images" / split / image.name, args.copy)
            link_or_copy(label, args.output / "labels" / split / label.name, args.copy)
        print(f"{split}: {len(split_pairs)}")


if __name__ == "__main__":
    main()

