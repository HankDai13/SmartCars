#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


YOLO_CLASSES = {
    0: "left",
    1: "right",
    2: "turnaround",
    3: "park",
    4: "person",
    5: "obstacle",
    6: "crosswalk",
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a static visual QA gallery for generated YOLO and LaneNet datasets."
    )
    parser.add_argument("--yolo-root", type=Path, default=Path("data/yolo"))
    parser.add_argument("--lanenet-root", type=Path, default=Path("data/lanenet_hnet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/dataset_visualization"))
    parser.add_argument("--per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Only include sources with this prefix. Can be passed multiple times, e.g. --source g04.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        default=[],
        help="Only include these split names. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
        help="Maximum displayed image width in generated JPEGs.",
    )
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def source_from_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_")
    for index, part in enumerate(parts):
        if part.isdigit() and len(part) == 6:
            return "_".join(parts[:index])
    return "_".join(parts[:2]) if len(parts) > 1 else parts[0]


def matches_filters(source: str, split: str | None, sources: list[str], splits: list[str]) -> bool:
    if sources and not any(source.startswith(item) for item in sources):
        return False
    if split is not None and splits and split not in splits:
        return False
    return True


def sample_by_source(items: list[dict[str, str]], per_source: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        grouped[item["source"]].append(item)
    sampled: list[dict[str, str]] = []
    for source in sorted(grouped):
        values = grouped[source]
        rng.shuffle(values)
        sampled.extend(values[:per_source])
    return sampled


def resize_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    return cv2.resize(
        image,
        (max_width, max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def draw_yolo_sample(image_path: Path, label_path: Path, max_width: int) -> tuple[np.ndarray, Counter[int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    counts: Counter[int] = Counter()
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        class_id_text, cx_text, cy_text, bw_text, bh_text = raw.split()
        class_id = int(class_id_text)
        cx = float(cx_text)
        cy = float(cy_text)
        bw = float(bw_text)
        bh = float(bh_text)
        x1 = int(round((cx - bw / 2) * width))
        y1 = int(round((cy - bh / 2) * height))
        x2 = int(round((cx + bw / 2) * width))
        y2 = int(round((cy + bh / 2) * height))
        counts[class_id] += 1
        color = class_color(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        label = YOLO_CLASSES.get(class_id, str(class_id))
        draw_label(image, label, x1, y1, color)
    return resize_max_width(image, max_width), counts


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (45, 220, 255),
        (80, 170, 255),
        (255, 190, 60),
        (70, 230, 120),
        (230, 90, 255),
        (255, 100, 90),
        (60, 255, 210),
    ]
    return palette[class_id % len(palette)]


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_text = max(text_height + 6, y)
    x = max(0, min(x, image.shape[1] - text_width - 4))
    cv2.rectangle(
        image,
        (x, y_text - text_height - baseline - 5),
        (x + text_width + 6, y_text + baseline),
        color,
        thickness=-1,
    )
    cv2.putText(image, text, (x + 3, y_text - 4), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def collect_yolo_samples(root: Path, sources: list[str], splits: list[str]) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    image_root = root / "images"
    if not image_root.exists():
        return samples
    flat_images = sorted(path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if flat_images:
        for image_path in flat_images:
            source = source_from_name(image_path.name)
            if not matches_filters(source, "flat", sources, splits):
                continue
            label_path = root / "labels" / f"{image_path.stem}.txt"
            if label_path.exists():
                samples.append(
                    {
                        "kind": "YOLO",
                        "source": source,
                        "split": "flat",
                        "image": str(image_path),
                        "label": str(label_path),
                    }
                )
        return samples

    for split_dir in sorted(image_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for image_path in sorted(path for path in split_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            source = source_from_name(image_path.name)
            if not matches_filters(source, split, sources, splits):
                continue
            label_path = root / "labels" / split / f"{image_path.stem}.txt"
            if label_path.exists():
                samples.append(
                    {
                        "kind": "YOLO",
                        "source": source,
                        "split": split,
                        "image": str(image_path),
                        "label": str(label_path),
                    }
                )
    return samples


def overlay_lanenet_sample(
    image_path: Path,
    binary_path: Path,
    instance_path: Path,
    max_width: int,
) -> tuple[np.ndarray, int, int]:
    image = cv2.imread(str(image_path))
    binary = cv2.imread(str(binary_path), cv2.IMREAD_GRAYSCALE)
    instance = cv2.imread(str(instance_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    if binary is None:
        raise FileNotFoundError(binary_path)
    if instance is None:
        raise FileNotFoundError(instance_path)

    overlay = image.copy()
    lane_pixels = binary > 0
    overlay[lane_pixels] = (0.45 * overlay[lane_pixels] + 0.55 * np.array([0, 255, 255])).astype(np.uint8)
    instance_scaled = ((instance.astype(np.uint16) * 47) % 255).astype(np.uint8)
    instance_color = cv2.applyColorMap(instance_scaled, cv2.COLORMAP_TURBO)
    instance_color[instance == 0] = (40, 28, 45)
    if image.shape[:2] != instance_color.shape[:2]:
        instance_color = cv2.resize(instance_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    joined = np.hstack([overlay, instance_color])
    return resize_max_width(joined, max_width), int(lane_pixels.sum()), int(instance.max())


def collect_lanenet_samples(root: Path, sources: list[str], splits: list[str]) -> list[dict[str, str]]:
    csv_path = root / "splits.csv"
    if not csv_path.exists():
        return []
    samples: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_name = row["image"]
            split = row["split"]
            source = source_from_name(image_name)
            if not matches_filters(source, split, sources, splits):
                continue
            stem = Path(image_name).stem
            image_path = root / "images" / image_name
            binary_path = root / "binary_masks" / f"{stem}.png"
            instance_path = root / "instance_masks" / f"{stem}.png"
            if image_path.exists() and binary_path.exists() and instance_path.exists():
                samples.append(
                    {
                        "kind": "LaneNet",
                        "source": source,
                        "split": split,
                        "image": str(image_path),
                        "binary": str(binary_path),
                        "instance": str(instance_path),
                    }
                )
    return samples


def write_html(output_dir: Path, cards: list[dict[str, str]], summary: list[str]) -> None:
    yolo_cards = [card for card in cards if card["kind"] == "YOLO"]
    lane_cards = [card for card in cards if card["kind"] == "LaneNet"]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dataset Visualization</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #16181d;
    }}
    body {{ margin: 0; }}
    header {{ padding: 24px 28px 12px; background: #ffffff; border-bottom: 1px solid #d8dde6; }}
    h1 {{ margin: 0 0 10px; font-size: 24px; font-weight: 700; }}
    h2 {{ margin: 26px 0 14px; font-size: 18px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 0; padding: 0; list-style: none; }}
    .summary li {{ background: #eef1f5; border: 1px solid #d8dde6; border-radius: 6px; padding: 6px 9px; font-size: 13px; }}
    main {{ padding: 0 28px 32px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; align-items: start; }}
    .card {{ background: #ffffff; border: 1px solid #d8dde6; border-radius: 8px; overflow: hidden; }}
    .card img {{ display: block; width: 100%; height: auto; background: #20242b; }}
    .meta {{ padding: 10px 12px; font-size: 12px; line-height: 1.5; color: #3c4350; }}
    .meta strong {{ color: #15181d; font-size: 13px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; }}
  </style>
</head>
<body>
  <header>
    <h1>Dataset Visualization</h1>
    <ul class="summary">
      {''.join(f'<li>{html.escape(item)}</li>' for item in summary)}
    </ul>
  </header>
  <main>
    <h2>YOLO</h2>
    <section class="grid">
      {''.join(render_card(card) for card in yolo_cards) or '<p>No YOLO samples matched.</p>'}
    </section>
    <h2>LaneNet</h2>
    <section class="grid">
      {''.join(render_card(card) for card in lane_cards) or '<p>No LaneNet samples matched.</p>'}
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def render_card(card: dict[str, str]) -> str:
    meta = [
        f"<strong>{html.escape(card['kind'])}</strong>",
        f"source: <code>{html.escape(card['source'])}</code>",
        f"split: <code>{html.escape(card['split'])}</code>",
        f"file: <code>{html.escape(Path(card['image']).name)}</code>",
    ]
    if card.get("detail"):
        meta.append(html.escape(card["detail"]))
    return f"""
<article class="card">
  <img src="{html.escape(card['preview'])}" alt="">
  <div class="meta">{'<br>'.join(meta)}</div>
</article>
"""


def safe_preview_name(index: int, sample: dict[str, str]) -> str:
    return f"{index:04d}_{sample['kind'].lower()}_{sample['source']}_{Path(sample['image']).stem}.jpg"


def write_visualizations(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.output_dir.exists() and not args.no_clean:
        shutil.rmtree(args.output_dir)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    yolo_samples = sample_by_source(
        collect_yolo_samples(args.yolo_root, args.source, args.split),
        args.per_source,
        args.seed,
    )
    lane_samples = sample_by_source(
        collect_lanenet_samples(args.lanenet_root, args.source, args.split),
        args.per_source,
        args.seed,
    )

    cards: list[dict[str, str]] = []
    for index, sample in enumerate(yolo_samples + lane_samples, start=1):
        preview_name = safe_preview_name(index, sample)
        preview_path = image_dir / preview_name
        if sample["kind"] == "YOLO":
            image, counts = draw_yolo_sample(Path(sample["image"]), Path(sample["label"]), args.max_width)
            sample["detail"] = "objects: " + ", ".join(
                f"{YOLO_CLASSES.get(class_id, str(class_id))}={count}" for class_id, count in sorted(counts.items())
            )
        else:
            image, lane_pixels, instance_count = overlay_lanenet_sample(
                Path(sample["image"]),
                Path(sample["binary"]),
                Path(sample["instance"]),
                args.max_width,
            )
            sample["detail"] = f"lane pixels: {lane_pixels}, instances: {instance_count}"
        cv2.imwrite(str(preview_path), image)
        sample["preview"] = str(Path("images") / preview_name)
        cards.append(sample)
    return cards


def format_distribution(samples: list[dict[str, str]], kind: str) -> str:
    counts = Counter(sample["source"] for sample in samples if sample["kind"] == kind)
    if not counts:
        return f"{kind}: 0"
    return f"{kind}: " + ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))


def main() -> None:
    args = parse_args()
    if args.per_source < 1:
        raise ValueError("--per-source must be positive")
    if args.max_width < 320:
        raise ValueError("--max-width must be at least 320")

    cards = write_visualizations(args)
    summary = [
        f"total previews: {len(cards)}",
        f"per source: {args.per_source}",
        f"filters: source={args.source or '*'}, split={args.split or '*'}",
        format_distribution(cards, "YOLO"),
        format_distribution(cards, "LaneNet"),
    ]
    write_html(args.output_dir, cards, summary)
    print(f"Wrote {len(cards)} previews to {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
