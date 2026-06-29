#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path

import cv2


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a flat YOLO dataset to LabelMe rectangle JSON files.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/relabel/park_yolo"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/relabel/park_labelme"))
    parser.add_argument("--class-name", default="park")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--embed-image-data",
        action="store_true",
        help="Embed base64 imageData in JSON. Disabled by default to keep files small.",
    )
    return parser.parse_args()


def read_image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    return height, width


def yolo_to_rectangle(line: str, width: int, height: int, class_name: str) -> dict | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
    except ValueError:
        return None
    x1 = max(0.0, (cx - bw / 2.0) * width)
    y1 = max(0.0, (cy - bh / 2.0) * height)
    x2 = min(float(width - 1), (cx + bw / 2.0) * width)
    y2 = min(float(height - 1), (cy + bh / 2.0) * height)
    return {
        "label": class_name,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
    }


def image_data(path: Path, embed: bool) -> str | None:
    if not embed:
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    args = parse_args()
    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in sorted((args.input_dir / "images").iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = args.input_dir / "labels" / f"{image_path.stem}.txt"
        height, width = read_image_shape(image_path)
        shapes = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                shape = yolo_to_rectangle(line, width, height, args.class_name)
                if shape is not None:
                    shapes.append(shape)
        target_image = args.output_dir / image_path.name
        shutil.copy2(image_path, target_image)
        data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": image_data(image_path, args.embed_image_data),
            "imageHeight": height,
            "imageWidth": width,
        }
        (args.output_dir / f"{image_path.stem}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        count += 1
    print(f"Converted {count} images to {args.output_dir}")


if __name__ == "__main__":
    main()
