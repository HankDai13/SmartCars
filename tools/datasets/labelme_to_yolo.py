#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe rectangle JSON files to flat YOLO labels.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/relabel/park_labelme"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/relabel/park_yolo_corrected"))
    parser.add_argument("--class-name", default="park")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to output-dir/images in addition to writing labels.",
    )
    return parser.parse_args()


def image_shape(image_path: Path, label_data: dict) -> tuple[int, int]:
    width = label_data.get("imageWidth")
    height = label_data.get("imageHeight")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return height, width
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]
    return h, w


def find_image(input_dir: Path, json_path: Path, label_data: dict) -> Path | None:
    candidates = []
    if label_data.get("imagePath"):
        candidates.append(input_dir / Path(str(label_data["imagePath"])).name)
    for suffix in IMAGE_SUFFIXES:
        candidates.append(input_dir / f"{json_path.stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def rectangle_to_yolo(shape: dict, width: int, height: int, class_id: int, class_name: str) -> str | None:
    if shape.get("label") != class_name:
        return None
    if shape.get("shape_type") != "rectangle":
        return None
    points = shape.get("points") or []
    if len(points) != 2:
        return None
    try:
        x_values = [float(points[0][0]), float(points[1][0])]
        y_values = [float(points[0][1]), float(points[1][1])]
    except (TypeError, ValueError, IndexError):
        return None
    x1 = max(0.0, min(x_values))
    x2 = min(float(width - 1), max(x_values))
    y1 = max(0.0, min(y_values))
    y2 = min(float(height - 1), max(y_values))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    label_out = args.output_dir / "labels"
    image_out = args.output_dir / "images"
    label_out.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        image_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    boxes = 0
    skipped_shapes = 0
    for json_path in sorted(args.input_dir.glob("*.json")):
        label_data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = find_image(args.input_dir, json_path, label_data)
        if image_path is None:
            raise FileNotFoundError(f"No image found for {json_path}")
        height, width = image_shape(image_path, label_data)
        lines = []
        for shape in label_data.get("shapes") or []:
            line = rectangle_to_yolo(shape, width, height, args.class_id, args.class_name)
            if line is None:
                skipped_shapes += 1
                continue
            lines.append(line)
        write_text(label_out / f"{json_path.stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
        if args.copy_images:
            shutil.copy2(image_path, image_out / image_path.name)
        converted += 1
        boxes += len(lines)

    write_text(args.output_dir / "classes.txt", f"{args.class_name}\n")
    write_text(
        args.output_dir / "data.yaml",
        f"path: .\ntrain: images\nval: images\nnames:\n  {args.class_id}: {args.class_name}\n",
    )
    print(f"Converted {converted} JSON files to {args.output_dir}")
    print(f"boxes: {boxes}, skipped non-matching/non-rectangle shapes: {skipped_shapes}")


if __name__ == "__main__":
    main()
