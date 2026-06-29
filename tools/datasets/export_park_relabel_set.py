#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_CLASSES = ["park"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all YOLO images containing the park class for manual relabeling."
    )
    parser.add_argument("--yolo-root", type=Path, default=Path("data/yolo"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/relabel/park_yolo"))
    parser.add_argument("--park-class-id", type=int, default=3)
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before exporting.")
    return parser.parse_args()


def source_from_stem(stem: str) -> str:
    parts = stem.split("_")
    for index, part in enumerate(parts):
        if part.isdigit() and len(part) == 6:
            return "_".join(parts[:index])
    return "_".join(parts[:2]) if len(parts) > 1 else parts[0]


def find_image(yolo_root: Path, split: str, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = yolo_root / "images" / split / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def park_only_labels(label_path: Path, park_class_id: int) -> list[str]:
    labels: list[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if class_id != park_class_id:
            continue
        labels.append("0 " + " ".join(parts[1:]))
    return labels


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    image_out = args.output_dir / "images"
    label_out = args.output_dir / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    by_source: Counter[str] = Counter()
    by_split: Counter[str] = Counter()

    for label_path in sorted((args.yolo_root / "labels").rglob("*.txt")):
        split = label_path.parent.name
        stem = label_path.stem
        labels = park_only_labels(label_path, args.park_class_id)
        if not labels:
            continue
        image_path = find_image(args.yolo_root, split, stem)
        if image_path is None:
            continue
        source = source_from_stem(stem)
        target_image = image_out / image_path.name
        target_label = label_out / f"{stem}.txt"
        shutil.copy2(image_path, target_image)
        write_text(target_label, "\n".join(labels) + "\n")
        rows.append(
            {
                "image": target_image.name,
                "label": target_label.name,
                "source": source,
                "split": split,
                "original_image": str(image_path),
                "original_label": str(label_path),
            }
        )
        by_source[source] += 1
        by_split[split] += 1

    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "label", "source", "split", "original_image", "original_label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_text(args.output_dir / "classes.txt", "\n".join(DEFAULT_CLASSES) + "\n")
    write_text(
        args.output_dir / "data.yaml",
        "path: .\ntrain: images\nval: images\nnames:\n  0: park\n",
    )
    write_text(
        args.output_dir / "README.md",
        """# Park Relabel Set

This directory contains every generated YOLO image that currently has a `park` object.

Relabeling rule:

- Keep exactly one class: `park`.
- Box the visible parking sign or ground `P` symbol tightly.
- Do not box the whole parking area, lane boundary, or green/gray background unless the symbol itself fills that region.
- Delete labels where no visible `P`/parking sign is present.

After relabeling, the files under `labels/` are the corrected YOLO labels.
Use `manifest.csv` to trace each image back to its generated source.
""",
    )

    print(f"Exported {len(rows)} park images to {args.output_dir}")
    print("by source:", ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))
    print("by split:", ", ".join(f"{k}={v}" for k, v in sorted(by_split.items())))


if __name__ == "__main__":
    main()
