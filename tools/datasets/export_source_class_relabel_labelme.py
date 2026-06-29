#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one source YOLO class from an imported image/label directory to LabelMe rectangles."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--source-class-id", type=int, required=True)
    parser.add_argument("--target-label", default="park")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def first_existing_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def read_image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return height, width


def yolo_line_to_rectangle(line: str, class_id: int, label: str, width: int, height: int) -> dict | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        source_class = int(parts[0])
        cx, cy, bw, bh = (float(value) for value in parts[1:])
    except ValueError:
        return None
    if source_class != class_id:
        return None
    x1 = max(0.0, (cx - bw / 2.0) * width)
    y1 = max(0.0, (cy - bh / 2.0) * height)
    x2 = min(float(width - 1), (cx + bw / 2.0) * width)
    y2 = min(float(height - 1), (cy + bh / 2.0) * height)
    return {
        "label": label,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
    }


def main() -> None:
    args = parse_args()
    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for label_path in sorted(args.label_dir.glob("*.txt")):
        if label_path.name == "classes.txt":
            continue
        image_path = first_existing_image(args.image_dir, label_path.stem)
        if image_path is None:
            continue
        height, width = read_image_shape(image_path)
        shapes = []
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            shape = yolo_line_to_rectangle(
                line=line,
                class_id=args.source_class_id,
                label=args.target_label,
                width=width,
                height=height,
            )
            if shape is not None:
                shapes.append(shape)
        if not shapes:
            continue

        output_name = f"{args.source_name}_{label_path.stem}{image_path.suffix.lower()}"
        target_image = args.output_dir / output_name
        shutil.copy2(image_path, target_image)
        data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": output_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        json_name = f"{Path(output_name).stem}.json"
        (args.output_dir / json_name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows.append(
            {
                "image": output_name,
                "json": json_name,
                "source_name": args.source_name,
                "source_image": str(image_path),
                "source_label": str(label_path),
            }
        )

    with (args.output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "json", "source_name", "source_image", "source_label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / "README.md").write_text(
        f"""# {args.source_name} Class {args.source_class_id} Relabel Set

Use exactly one label: `{args.target_label}`.

Only keep rectangles around visible `P`, circled `P`, or parking signs.
Delete rectangles and save empty JSON when only slot boundary lines or parking area background are visible.
""",
        encoding="utf-8",
    )
    print(f"Exported {len(rows)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
