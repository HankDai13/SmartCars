#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Iterable, TypeVar

import cv2
import numpy as np


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
T = TypeVar("T")

TARGET_NAMES = {
    0: "left",
    1: "right",
    2: "turnaround",
    3: "park",
    4: "person",
    5: "obstacle",
    6: "crosswalk",
}

TARGET_BY_NAME = {name: idx for idx, name in TARGET_NAMES.items()}

SOURCE_CLASS_MAP: dict[str, int | None] = {
    "left": TARGET_BY_NAME["left"],
    "leftdw": TARGET_BY_NAME["left"],
    "right": TARGET_BY_NAME["right"],
    "return": TARGET_BY_NAME["turnaround"],
    "back": TARGET_BY_NAME["turnaround"],
    "turnaround": TARGET_BY_NAME["turnaround"],
    "park": TARGET_BY_NAME["park"],
    "person": TARGET_BY_NAME["person"],
    "obs": TARGET_BY_NAME["obstacle"],
    "obs1": TARGET_BY_NAME["obstacle"],
    "obs2": TARGET_BY_NAME["obstacle"],
    "obstacle": TARGET_BY_NAME["obstacle"],
    "sideway": TARGET_BY_NAME["crosswalk"],
    "crosswalk": TARGET_BY_NAME["crosswalk"],
    "stop": None,
}

SOURCE_CLASS_OVERRIDES: dict[str, dict[str, int | None]] = {
    "g01_yolo_09": {
        "obs1": TARGET_BY_NAME["park"],
    },
    "g01_yolo_10": {
        "obs1": TARGET_BY_NAME["park"],
    },
    "g01_yolo_14": {
        "obs1": TARGET_BY_NAME["park"],
    },
}

DEFAULT_PARK_RELABEL_DIRS = (
    Path("data/relabel/park_yolo_corrected"),
    Path("data/relabel/park_yolo_g01_yolo_10_corrected"),
)


@dataclass(frozen=True)
class ImageRef:
    path: Path | None = None
    zip_path: Path | None = None
    zip_member: str | None = None

    def read_bytes(self) -> bytes:
        if self.path is not None:
            return self.path.read_bytes()
        if self.zip_path is None or self.zip_member is None:
            raise ValueError("ImageRef is missing both file and zip source.")
        with zipfile.ZipFile(self.zip_path) as zf:
            return zf.read(self.zip_member)

    def display(self) -> str:
        if self.path is not None:
            return str(self.path)
        return f"{self.zip_path}!{self.zip_member}"


class ImageReader:
    def __init__(self) -> None:
        self._zips: dict[Path, zipfile.ZipFile] = {}

    def read(self, ref: ImageRef) -> bytes:
        if ref.path is not None:
            return ref.path.read_bytes()
        if ref.zip_path is None or ref.zip_member is None:
            raise ValueError("ImageRef is missing both file and zip source.")
        zf = self._zips.get(ref.zip_path)
        if zf is None:
            zf = zipfile.ZipFile(ref.zip_path)
            self._zips[ref.zip_path] = zf
        return zf.read(ref.zip_member)

    def close(self) -> None:
        for zf in self._zips.values():
            zf.close()
        self._zips.clear()

    def __enter__(self) -> "ImageReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass
class YoloSample:
    source: str
    original_stem: str
    image_ref: ImageRef
    image_ext: str
    image_sha1: str
    label_lines: list[str]
    mapped_counts: Counter[int]
    skipped_objects: Counter[str]


@dataclass
class LaneShape:
    label: str
    shape_type: str
    points: list[tuple[int, int]]


@dataclass
class LaneSample:
    source: str
    original_stem: str
    image_ref: ImageRef
    image_ext: str
    image_sha1: str
    shapes: list[LaneShape]


class ImportStats:
    def __init__(self) -> None:
        self.source_seen: Counter[str] = Counter()
        self.source_accepted: Counter[str] = Counter()
        self.source_skipped: dict[str, Counter[str]] = defaultdict(Counter)
        self.yolo_class_counts: Counter[int] = Counter()
        self.yolo_skipped_objects: dict[str, Counter[str]] = defaultdict(Counter)
        self.yolo_source_names: dict[str, list[str]] = {}
        self.lane_line_labels: dict[str, Counter[str]] = defaultdict(Counter)
        self.notes: list[str] = []
        self.park_relabel_seen = 0
        self.park_relabel_applied = 0
        self.park_relabel_removed = 0

    def skip(self, source: str, reason: str, count: int = 1) -> None:
        if count == 0:
            return
        self.source_skipped[source][reason] += count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified YOLO and LaneNet datasets from data/imported."
    )
    parser.add_argument("--import-root", type=Path, default=Path("data/imported"))
    parser.add_argument("--output-yolo", type=Path, default=Path("data/yolo"))
    parser.add_argument(
        "--output-lanenet", type=Path, default=Path("data/lanenet_hnet")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/imported_dataset_report.md")
    )
    parser.add_argument(
        "--park-relabel-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional corrected flat YOLO relabel set for park samples. "
            "Repeat to apply multiple relabel directories."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument(
        "--lane-line-width",
        type=int,
        default=5,
        help="Pixel thickness used when rasterizing LabelMe lane line annotations.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove previously generated output subdirectories before writing.",
    )
    return parser.parse_args()


def ensure_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def clean_outputs(yolo_root: Path, lane_root: Path) -> None:
    for path in [
        yolo_root / "images",
        yolo_root / "labels",
        lane_root / "images",
        lane_root / "binary_masks",
        lane_root / "instance_masks",
    ]:
        if path.exists():
            shutil.rmtree(path)
    split_csv = lane_root / "splits.csv"
    if split_csv.exists():
        split_csv.unlink()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_classes(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in read_text(path).splitlines() if line.strip()]


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def first_existing_image(root: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def portable_path_name(value: str) -> str:
    return PureWindowsPath(value).name if "\\" in value else Path(value).name


def sanitize_source_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "sample"


def split_samples(samples: list[T], seed: int, train: float, val: float) -> dict[str, list[T]]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    n_train = int(total * train)
    n_val = int(total * val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def output_name(sample_source: str, index: int, sha1: str, ext: str) -> str:
    return f"{sanitize_source_name(sample_source)}_{index:06d}_{sha1[:10]}{ext.lower()}"


def parse_yolo_label(
    label_text: str,
    class_names: list[str],
    stats: ImportStats,
    source: str,
) -> tuple[list[str], Counter[int], Counter[str], bool]:
    label_lines: list[str] = []
    mapped_counts: Counter[int] = Counter()
    skipped_objects: Counter[str] = Counter()
    had_format_error = False

    for line_no, raw in enumerate(label_text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            had_format_error = True
            skipped_objects[f"bad_line:{line_no}"] += 1
            continue
        try:
            source_class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            had_format_error = True
            skipped_objects[f"bad_numeric:{line_no}"] += 1
            continue
        if any(value < 0.0 or value > 1.0 for value in coords):
            had_format_error = True
            skipped_objects[f"bad_bbox:{line_no}"] += 1
            continue

        if 0 <= source_class_id < len(class_names):
            source_name = normalize_class_name(class_names[source_class_id])
        else:
            source_name = f"class_{source_class_id}"
        target_class = SOURCE_CLASS_OVERRIDES.get(source, {}).get(
            source_name, SOURCE_CLASS_MAP.get(source_name)
        )
        if target_class is None:
            skipped_objects[source_name] += 1
            stats.yolo_skipped_objects[source][source_name] += 1
            continue

        label_lines.append(
            f"{target_class} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
        )
        mapped_counts[target_class] += 1
        stats.yolo_class_counts[target_class] += 1

    return label_lines, mapped_counts, skipped_objects, had_format_error


def add_yolo_sample(
    samples: list[YoloSample],
    seen_hashes: set[str],
    stats: ImportStats,
    source: str,
    original_stem: str,
    image_ref: ImageRef,
    image_ext: str,
    label_text: str,
    class_names: list[str],
    image_data: bytes | None = None,
) -> None:
    stats.source_seen[source] += 1
    data = image_data if image_data is not None else image_ref.read_bytes()
    image_sha1 = sha1_bytes(data)
    if image_sha1 in seen_hashes:
        stats.skip(source, "duplicate_image")
        return

    label_lines, mapped_counts, skipped_objects, had_format_error = parse_yolo_label(
        label_text, class_names, stats, source
    )
    if had_format_error and not label_lines:
        stats.skip(source, "invalid_label_file")
        return
    if not label_lines:
        stats.skip(source, "empty_after_mapping")
        return

    seen_hashes.add(image_sha1)
    stats.source_accepted[source] += 1
    samples.append(
        YoloSample(
            source=source,
            original_stem=original_stem,
            image_ref=image_ref,
            image_ext=image_ext,
            image_sha1=image_sha1,
            label_lines=label_lines,
            mapped_counts=mapped_counts,
            skipped_objects=skipped_objects,
        )
    )


def collect_yolo_directory(
    samples: list[YoloSample],
    seen_hashes: set[str],
    stats: ImportStats,
    source: str,
    image_dir: Path,
    label_dir: Path,
    class_names: list[str],
) -> None:
    if not image_dir.exists() or not label_dir.exists():
        stats.skip(source, "missing_dir")
        return
    stats.yolo_source_names[source] = class_names
    for label_path in sorted(label_dir.glob("*.txt")):
        if label_path.name == "classes.txt":
            continue
        image_path = first_existing_image(image_dir, label_path.stem)
        if image_path is None:
            stats.source_seen[source] += 1
            stats.skip(source, "missing_image")
            continue
        add_yolo_sample(
            samples=samples,
            seen_hashes=seen_hashes,
            stats=stats,
            source=source,
            original_stem=label_path.stem,
            image_ref=ImageRef(path=image_path),
            image_ext=image_path.suffix,
            label_text=read_text(label_path),
            class_names=class_names,
        )


def collect_yolo_ultralytics_cache(
    samples: list[YoloSample],
    seen_hashes: set[str],
    stats: ImportStats,
    source: str,
    image_dir: Path,
    cache_paths: Iterable[Path],
    class_names: list[str],
) -> None:
    if not image_dir.exists():
        stats.skip(source, "missing_dir")
        return
    stats.yolo_source_names[source] = class_names
    for cache_path in sorted(cache_paths):
        if not cache_path.exists():
            stats.skip(source, "missing_cache")
            continue
        try:
            cache_data = np.load(cache_path, allow_pickle=True).item()
        except (OSError, ValueError, AttributeError):
            stats.skip(source, "bad_cache")
            continue
        labels = cache_data.get("labels") or []
        for item in labels:
            image_name = portable_path_name(str(item.get("im_file", "")))
            stem = Path(image_name).stem
            image_path = first_existing_image(image_dir, stem)
            if image_path is None:
                stats.source_seen[source] += 1
                stats.skip(source, "missing_image")
                continue
            cls = item.get("cls")
            boxes = item.get("bboxes")
            normalized = bool(item.get("normalized"))
            bbox_format = str(item.get("bbox_format", "")).lower()
            if cls is None or boxes is None or not normalized or bbox_format != "xywh":
                stats.source_seen[source] += 1
                stats.skip(source, "unsupported_cache_label")
                continue
            cls_array = np.asarray(cls).reshape(-1)
            box_array = np.asarray(boxes, dtype=float)
            if box_array.ndim != 2 or box_array.shape[1] != 4 or len(cls_array) != len(box_array):
                stats.source_seen[source] += 1
                stats.skip(source, "bad_cache_label")
                continue
            label_lines = [
                f"{int(class_id)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                for class_id, box in zip(cls_array, box_array, strict=True)
            ]
            add_yolo_sample(
                samples=samples,
                seen_hashes=seen_hashes,
                stats=stats,
                source=source,
                original_stem=stem,
                image_ref=ImageRef(path=image_path),
                image_ext=image_path.suffix,
                label_text="\n".join(label_lines),
                class_names=class_names,
            )


def collect_yolo_zip(
    samples: list[YoloSample],
    seen_hashes: set[str],
    stats: ImportStats,
    source: str,
    zip_path: Path,
    image_prefix: str,
    label_prefix: str,
    class_names: list[str],
) -> None:
    if not zip_path.exists():
        stats.skip(source, "missing_zip")
        return
    stats.yolo_source_names[source] = class_names
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        image_by_stem = {
            Path(name).stem: name
            for name in names
            if name.startswith(image_prefix)
            and Path(name).suffix.lower() in IMAGE_SUFFIXES
        }
        labels = sorted(
            name
            for name in names
            if name.startswith(label_prefix) and name.lower().endswith(".txt")
        )
        for label_member in labels:
            if Path(label_member).name == "classes.txt":
                continue
            stem = Path(label_member).stem
            image_member = image_by_stem.get(stem)
            if image_member is None:
                stats.source_seen[source] += 1
                stats.skip(source, "missing_image")
                continue
            add_yolo_sample(
                samples=samples,
                seen_hashes=seen_hashes,
                stats=stats,
                source=source,
                original_stem=stem,
                image_ref=ImageRef(zip_path=zip_path, zip_member=image_member),
                image_ext=Path(image_member).suffix,
                label_text=zf.read(label_member).decode("utf-8", errors="replace"),
                class_names=class_names,
                image_data=zf.read(image_member),
            )


def labelme_shapes(data: dict) -> list[LaneShape]:
    lane_shapes: list[LaneShape] = []
    shapes = data.get("shapes") or []
    for shape in shapes:
        shape_type = str(shape.get("shape_type") or "polygon")
        if shape_type not in {"line", "linestrip", "polygon"}:
            continue
        points = shape.get("points") or []
        min_points = 2 if shape_type in {"line", "linestrip"} else 3
        if len(points) < min_points:
            continue
        rounded_points: list[tuple[int, int]] = []
        try:
            for point in points:
                rounded_points.append((int(round(point[0])), int(round(point[1]))))
        except (TypeError, ValueError, IndexError):
            continue
        lane_shapes.append(
            LaneShape(
                label=str(shape.get("label", "line")),
                shape_type=shape_type,
                points=rounded_points,
            )
        )
    return lane_shapes


def add_lane_sample(
    samples: list[LaneSample],
    seen_hashes: set[str],
    stats: ImportStats,
    source: str,
    original_stem: str,
    image_ref: ImageRef,
    image_ext: str,
    label_data: dict,
    image_data: bytes | None = None,
) -> None:
    stats.source_seen[source] += 1
    shapes = labelme_shapes(label_data)
    if not shapes:
        stats.skip(source, "no_lane_shapes")
        return
    data = image_data if image_data is not None else image_ref.read_bytes()
    image_sha1 = sha1_bytes(data)
    if image_sha1 in seen_hashes:
        stats.skip(source, "duplicate_image")
        return
    seen_hashes.add(image_sha1)
    stats.source_accepted[source] += 1
    for shape in shapes:
        stats.lane_line_labels[source][f"{shape.shape_type}:{shape.label}"] += 1
    samples.append(
        LaneSample(
            source=source,
            original_stem=original_stem,
            image_ref=image_ref,
            image_ext=image_ext,
            image_sha1=image_sha1,
            shapes=shapes,
        )
    )


def collect_lane_labelme_directory(
    samples: list[LaneSample],
    seen_hashes: set[str],
    stats: ImportStats,
    source: str,
    image_dir: Path,
    label_dir: Path,
) -> None:
    if not image_dir.exists() or not label_dir.exists():
        stats.skip(source, "missing_dir")
        return
    for label_path in sorted(label_dir.glob("*.json")):
        try:
            label_data = json.loads(read_text(label_path))
        except json.JSONDecodeError:
            stats.source_seen[source] += 1
            stats.skip(source, "bad_json")
            continue

        image_path = first_existing_image(image_dir, label_path.stem)
        if image_path is None and label_data.get("imagePath"):
            image_path = first_existing_image(image_dir, Path(label_data["imagePath"]).stem)
        if image_path is None:
            stats.source_seen[source] += 1
            stats.skip(source, "missing_image")
            continue
        add_lane_sample(
            samples=samples,
            seen_hashes=seen_hashes,
            stats=stats,
            source=source,
            original_stem=label_path.stem,
            image_ref=ImageRef(path=image_path),
            image_ext=image_path.suffix,
            label_data=label_data,
        )


def collect_lane_labelme_zip(
    samples: list[LaneSample],
    seen_hashes: set[str],
    stats: ImportStats,
    source: str,
    zip_path: Path,
    image_prefix: str,
    label_prefix: str,
) -> None:
    if not zip_path.exists():
        stats.skip(source, "missing_zip")
        return
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        image_by_stem = {
            Path(name).stem: name
            for name in names
            if name.startswith(image_prefix)
            and Path(name).suffix.lower() in IMAGE_SUFFIXES
        }
        labels = sorted(
            name
            for name in names
            if name.startswith(label_prefix) and name.lower().endswith(".json")
        )
        for label_member in labels:
            stem = Path(label_member).stem
            image_member = image_by_stem.get(stem)
            if image_member is None:
                stats.source_seen[source] += 1
                stats.skip(source, "missing_image")
                continue
            try:
                label_data = json.loads(
                    zf.read(label_member).decode("utf-8", errors="replace")
                )
            except json.JSONDecodeError:
                stats.source_seen[source] += 1
                stats.skip(source, "bad_json")
                continue
            add_lane_sample(
                samples=samples,
                seen_hashes=seen_hashes,
                stats=stats,
                source=source,
                original_stem=stem,
                image_ref=ImageRef(zip_path=zip_path, zip_member=image_member),
                image_ext=Path(image_member).suffix,
                label_data=label_data,
                image_data=zf.read(image_member),
            )


def iter_import_paths(root: Path) -> Iterable[Path]:
    ignored_dirs = {".venv", "venv", "env", "__pycache__", ".git", ".idea", ".vscode"}
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda path: path.name)
        except OSError:
            continue
        for child in children:
            yield child
            if child.is_dir() and child.name not in ignored_dirs:
                stack.append(child)


def find_first(root: Path, predicate) -> Path | None:
    for path in iter_import_paths(root):
        if predicate(path):
            return path
    return None


def find_all(root: Path, predicate) -> list[Path]:
    return [path for path in iter_import_paths(root) if predicate(path)]


def top_level_dir(import_root: Path, token: str) -> Path | None:
    for path in sorted(import_root.iterdir(), key=lambda item: item.name):
        if path.is_dir() and token in path.name:
            return path
    return None


def collect_all(import_root: Path) -> tuple[list[YoloSample], list[LaneSample], ImportStats]:
    stats = ImportStats()
    yolo_samples: list[YoloSample] = []
    lane_samples: list[LaneSample] = []
    seen_yolo_hashes: set[str] = set()
    seen_lane_hashes: set[str] = set()

    g01_root = top_level_dir(import_root, "01组")
    g04_root = top_level_dir(import_root, "4组")
    g16_root = top_level_dir(import_root, "16组")
    g23_root = top_level_dir(import_root, "23组--杨锐")
    liu_root = top_level_dir(import_root, "刘航")

    g01_yolo_root = find_first(
        g01_root or import_root,
        lambda p: p.is_dir() and p.name == "yolo_dataset" and "01组" in str(p),
    )
    if g01_yolo_root is not None:
        for label_dir in sorted(p for p in g01_yolo_root.iterdir() if p.is_dir()):
            if not label_dir.name.endswith("label"):
                continue
            prefix = label_dir.name[: -len("label")]
            image_dir = g01_yolo_root / f"{prefix}images"
            class_options = [
                read_classes(label_dir / "classes.txt"),
                read_classes(image_dir / "classes.txt"),
                ["left", "right", "return", "obs1", "obs2", "leftdw"],
            ]
            class_names = max(class_options, key=len)
            collect_yolo_directory(
                yolo_samples,
                seen_yolo_hashes,
                stats,
                source=f"g01_yolo_{prefix}",
                image_dir=image_dir,
                label_dir=label_dir,
                class_names=class_names,
            )
    else:
        stats.notes.append("Missing group 01 yolo_dataset directory.")

    g23_sign_root = find_first(
        g23_root or import_root,
        lambda p: p.is_dir()
        and p.name == "sign"
        and "23组" in str(p)
        and "车道线数据及标注结果" in str(p),
    )
    if g23_sign_root is not None:
        class_names = read_classes(g23_sign_root / "classes.txt")
        collect_yolo_directory(
            yolo_samples,
            seen_yolo_hashes,
            stats,
            source="g23_sign",
            image_dir=g23_sign_root,
            label_dir=g23_sign_root,
            class_names=class_names,
        )
        stats.notes.append(
            "Skipped group 23 YOLO training copies under 4.模型训练/yolo to avoid duplicate sign data."
        )
    else:
        stats.notes.append("Missing group 23 sign directory.")

    g16_zips = {
        "g16_yolo_road1": (
            find_first(g16_root or import_root, lambda p: p.is_file() and p.name == "车道线数据标注结果1.zip"),
            "road1/images01/",
            "road1/images01_label/",
        ),
        "g16_yolo_road2": (
            find_first(g16_root or import_root, lambda p: p.is_file() and p.name == "车道线数据标注结果2.zip"),
            "road2/road2/",
            "road2/road2标注/",
        ),
    }
    # The group 16 zip files store classes.txt as numeric ids instead of semantic
    # names. Visual inspection of the raw annotations shows this source order.
    g16_classes = ["turnaround", "crosswalk", "left", "right", "park"]
    for source, (zip_path, image_prefix, label_prefix) in g16_zips.items():
        if zip_path is None:
            stats.skip(source, "missing_zip")
            continue
        collect_yolo_zip(
            yolo_samples,
            seen_yolo_hashes,
            stats,
            source=source,
            zip_path=zip_path,
            image_prefix=image_prefix,
            label_prefix=label_prefix,
            class_names=g16_classes,
        )
    if find_first(g16_root or import_root, lambda p: p.is_file() and p.name == "车道线数据标注结果1(1).zip"):
        stats.notes.append("Skipped 16组 车道线数据标注结果1(1).zip because it duplicates 结果1.zip.")

    g04_landmark_root = find_first(
        g04_root or import_root,
        lambda p: p.is_dir()
        and p.name == "地标数据"
        and "4组" in str(p)
        and "车道线训练数据及标注结果" in str(p),
    )
    if g04_landmark_root is not None:
        collect_yolo_ultralytics_cache(
            yolo_samples,
            seen_yolo_hashes,
            stats,
            source="g04_landmark",
            image_dir=g04_landmark_root / "images" / "train",
            cache_paths=(g04_landmark_root / "labels").glob("*.cache"),
            class_names=["park", "stop", "right", "back", "sideway", "left"],
        )
        stats.notes.append(
            "Imported group 04 YOLO labels from Ultralytics .cache files because raw .txt labels were not included."
        )
    else:
        stats.notes.append("Missing group 04 landmark data directory.")

    g01_lane_label_dir = find_first(
        g01_root or import_root,
        lambda p: p.is_dir()
        and p.name == "all_label"
        and "lanenet_data" in str(p)
        and "01组" in str(p),
    )
    if g01_lane_label_dir is not None:
        collect_lane_labelme_directory(
            lane_samples,
            seen_lane_hashes,
            stats,
            source="g01_lanenet",
            image_dir=g01_lane_label_dir.parent / "all_image",
            label_dir=g01_lane_label_dir,
        )
    else:
        stats.notes.append("Missing group 01 lanenet_data/all_label directory.")

    g23_roadline_root = find_first(
        g23_root or import_root,
        lambda p: p.is_dir()
        and p.name == "roadline"
        and "23组" in str(p)
        and "车道线数据及标注结果" in str(p),
    )
    if g23_roadline_root is not None:
        collect_lane_labelme_directory(
            lane_samples,
            seen_lane_hashes,
            stats,
            source="g23_roadline",
            image_dir=g23_roadline_root,
            label_dir=g23_roadline_root,
        )
    else:
        stats.notes.append("Missing group 23 roadline directory.")

    g04_roadline_root = find_first(
        g04_root or import_root,
        lambda p: p.is_dir()
        and p.name == "车道线数据"
        and "4组" in str(p)
        and "车道线训练数据及标注结果" in str(p),
    )
    if g04_roadline_root is not None:
        collect_lane_labelme_directory(
            lane_samples,
            seen_lane_hashes,
            stats,
            source="g04_lanenet",
            image_dir=g04_roadline_root / "images",
            label_dir=g04_roadline_root / "labels",
        )
    else:
        stats.notes.append("Missing group 04 roadline data directory.")

    liu_lane_zips = {
        "liu_images01": (
            find_first(liu_root or import_root, lambda p: p.is_file() and p.name == "images01.zip"),
            "images01/images/",
            "images01/labels/",
        ),
        "liu_images02": (
            find_first(liu_root or import_root, lambda p: p.is_file() and p.name == "images02.zip"),
            "images02/images/",
            "images02/labels/",
        ),
    }
    for source, (zip_path, image_prefix, label_prefix) in liu_lane_zips.items():
        if zip_path is None:
            stats.skip(source, "missing_zip")
            continue
        collect_lane_labelme_zip(
            lane_samples,
            seen_lane_hashes,
            stats,
            source=source,
            zip_path=zip_path,
            image_prefix=image_prefix,
            label_prefix=label_prefix,
        )
    if find_first(liu_root or import_root, lambda p: p.is_file() and p.suffix.lower() == ".7z"):
        stats.notes.append("Skipped .7z archives; no 7z reader is required for the usable labeled data found.")

    return yolo_samples, lane_samples, stats


def relabel_keys_from_stem(stem: str) -> list[tuple[str, str]]:
    parts = stem.split("_")
    keys: list[tuple[str, str]] = []
    for index, part in enumerate(parts):
        if part.isdigit() and len(part) == 6 and index + 1 < len(parts):
            keys.append(("_".join(parts[:index]), f"sha:{parts[index + 1]}"))
            break
    for index in range(1, len(parts)):
        source = "_".join(parts[:index])
        original_stem = "_".join(parts[index:])
        if source and original_stem:
            keys.append((source, f"stem:{original_stem}"))
    return keys


def read_corrected_park_labels(
    relabel_dirs: Iterable[Path],
) -> tuple[dict[tuple[str, str], list[str]], list[Path], list[Path], int]:
    corrected: dict[tuple[str, str], list[str]] = {}
    found_dirs: list[Path] = []
    missing_dirs: list[Path] = []
    invalid_files = 0

    for relabel_dir in relabel_dirs:
        labels_dir = relabel_dir / "labels"
        if not labels_dir.exists():
            missing_dirs.append(relabel_dir)
            continue
        found_dirs.append(relabel_dir)
        for label_path in sorted(labels_dir.glob("*.txt")):
            keys = relabel_keys_from_stem(label_path.stem)
            if not keys:
                invalid_files += 1
                continue
            lines: list[str] = []
            for raw in read_text(label_path).splitlines():
                if not raw.strip():
                    continue
                parts = raw.split()
                if len(parts) != 5:
                    continue
                try:
                    source_class = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    continue
                if source_class != 0:
                    continue
                if any(value < 0.0 or value > 1.0 for value in coords):
                    continue
                lines.append(
                    f"{TARGET_BY_NAME['park']} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
                )
            for key in keys:
                corrected[key] = lines
    return corrected, found_dirs, missing_dirs, invalid_files


def apply_corrected_park_labels(
    samples: list[YoloSample],
    stats: ImportStats,
    relabel_dirs: Iterable[Path],
) -> list[YoloSample]:
    relabel_dirs = list(relabel_dirs)
    corrected, found_dirs, missing_dirs, invalid_files = read_corrected_park_labels(relabel_dirs)
    for relabel_dir in missing_dirs:
        stats.notes.append(f"No corrected park relabel labels found under {relabel_dir}.")
    if invalid_files:
        stats.notes.append(f"Skipped {invalid_files} corrected park relabel files with unrecognized names.")
    if not corrected:
        return samples

    updated: list[YoloSample] = []
    for sample in samples:
        source = sanitize_source_name(sample.source)
        sample_keys = [
            (source, f"sha:{sample.image_sha1[:10]}"),
            (source, f"stem:{sample.original_stem}"),
        ]
        corrected_lines = next((corrected[key] for key in sample_keys if key in corrected), None)
        if corrected_lines is None:
            updated.append(sample)
            continue

        stats.park_relabel_seen += 1
        non_park_lines = [
            line for line in sample.label_lines if int(line.split()[0]) != TARGET_BY_NAME["park"]
        ]
        new_label_lines = non_park_lines + corrected_lines
        if not new_label_lines:
            stats.park_relabel_removed += 1
            stats.skip(sample.source, "removed_by_park_relabel")
            continue

        sample.label_lines = new_label_lines
        sample.mapped_counts = Counter(int(line.split()[0]) for line in new_label_lines)
        stats.park_relabel_applied += 1
        updated.append(sample)

    if stats.park_relabel_seen:
        stats.notes.append(
            f"Applied corrected park relabel sets from {', '.join(str(path) for path in found_dirs)}: matched {stats.park_relabel_seen}, kept {stats.park_relabel_applied}, removed {stats.park_relabel_removed} empty samples."
        )
    return updated


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_yolo_dataset(
    samples: list[YoloSample],
    output_root: Path,
    seed: int,
    train: float,
    val: float,
) -> dict[str, int]:
    splits = split_samples(samples, seed, train, val)
    counts: dict[str, int] = {}
    global_index = 0
    with ImageReader() as reader:
        for split, split_samples_ in splits.items():
            counts[split] = len(split_samples_)
            for sample in split_samples_:
                global_index += 1
                image_name = output_name(sample.source, global_index, sample.image_sha1, sample.image_ext)
                label_name = f"{Path(image_name).stem}.txt"
                image_data = reader.read(sample.image_ref)
                image = decode_image(image_data)
                if image is None or image.size == 0:
                    raise FileNotFoundError(f"Unreadable image: {sample.image_ref.display()}")
                write_bytes(output_root / "images" / split / image_name, image_data)
                write_text(
                    output_root / "labels" / split / label_name,
                    "\n".join(sample.label_lines) + ("\n" if sample.label_lines else ""),
                )
    return counts


def render_lane_masks(
    sample: LaneSample,
    image_data: bytes,
    binary_path: Path,
    instance_path: Path,
    line_width: int,
) -> None:
    image = decode_image(image_data)
    if image is None:
        raise FileNotFoundError(sample.image_ref.display())
    height, width = image.shape[:2]
    binary = np.zeros((height, width), dtype=np.uint8)
    instance = np.zeros((height, width), dtype=np.uint8)
    for lane_id, shape in enumerate(sample.shapes, start=1):
        points = np.array(shape.points, dtype=np.int32)
        if shape.shape_type == "polygon":
            cv2.fillPoly(binary, [points], color=255)
            cv2.fillPoly(instance, [points], color=lane_id)
        else:
            cv2.polylines(binary, [points], isClosed=False, color=255, thickness=line_width)
            cv2.polylines(instance, [points], isClosed=False, color=lane_id, thickness=line_width)
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    instance_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(binary_path), binary)
    cv2.imwrite(str(instance_path), instance)


def write_lanenet_dataset(
    samples: list[LaneSample],
    output_root: Path,
    seed: int,
    train: float,
    val: float,
    line_width: int,
) -> dict[str, int]:
    splits = split_samples(samples, seed, train, val)
    counts: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    global_index = 0
    with ImageReader() as reader:
        for split, split_samples_ in splits.items():
            counts[split] = len(split_samples_)
            for sample in split_samples_:
                global_index += 1
                image_name = output_name(sample.source, global_index, sample.image_sha1, sample.image_ext)
                mask_name = f"{Path(image_name).stem}.png"
                image_data = reader.read(sample.image_ref)
                write_bytes(output_root / "images" / image_name, image_data)
                render_lane_masks(
                    sample=sample,
                    image_data=image_data,
                    binary_path=output_root / "binary_masks" / mask_name,
                    instance_path=output_root / "instance_masks" / mask_name,
                    line_width=line_width,
                )
                rows.append({"image": image_name, "split": split})

    split_csv = output_root / "splits.csv"
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    with split_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return counts


def format_counter(counter: Counter) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in sorted(counter.items(), key=lambda item: str(item[0])))


def write_report(
    report_path: Path,
    stats: ImportStats,
    yolo_samples: list[YoloSample],
    lane_samples: list[LaneSample],
    yolo_split_counts: dict[str, int],
    lane_split_counts: dict[str, int],
    line_width: int,
) -> None:
    yolo_objects = Counter()
    yolo_empty = 0
    for sample in yolo_samples:
        yolo_objects.update(sample.mapped_counts)
        if not sample.label_lines:
            yolo_empty += 1
    lane_shapes = Counter()
    for sample in lane_samples:
        lane_shapes[sample.source] += len(sample.shapes)

    lines: list[str] = []
    lines.append("# Imported Dataset Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Target YOLO Classes")
    lines.append("")
    lines.append("| id | name |")
    lines.append("| --- | --- |")
    for class_id, name in TARGET_NAMES.items():
        lines.append(f"| {class_id} | {name} |")
    lines.append("")
    lines.append("## YOLO Summary")
    lines.append("")
    lines.append(f"- accepted images: {len(yolo_samples)}")
    lines.append(f"- empty labels after mapping: {yolo_empty}")
    lines.append(f"- split counts: {format_counter(Counter(yolo_split_counts))}")
    if stats.park_relabel_seen:
        lines.append(
            f"- corrected park relabel: matched {stats.park_relabel_seen}, kept {stats.park_relabel_applied}, removed empty {stats.park_relabel_removed}"
        )
    lines.append(
        "- class distribution: "
        + ", ".join(
            f"{class_id}:{TARGET_NAMES[class_id]}={yolo_objects[class_id]}"
            for class_id in TARGET_NAMES
        )
    )
    lines.append("")
    lines.append("| source | seen labels | accepted images | skipped | source classes | skipped objects |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for source in sorted(set(stats.source_seen) | set(stats.source_accepted)):
        if not source.startswith(("g01_yolo", "g23_sign", "g16_yolo", "g04_landmark")):
            continue
        lines.append(
            f"| {source} | {stats.source_seen[source]} | {stats.source_accepted[source]} | "
            f"{format_counter(stats.source_skipped[source])} | "
            f"{', '.join(stats.yolo_source_names.get(source, [])) or '-'} | "
            f"{format_counter(stats.yolo_skipped_objects[source])} |"
        )
    lines.append("")
    lines.append("YOLO class remapping used:")
    lines.append("")
    for source_name, target_id in sorted(SOURCE_CLASS_MAP.items()):
        target = "skip" if target_id is None else f"{target_id}:{TARGET_NAMES[target_id]}"
        lines.append(f"- `{source_name}` -> {target}")
    if SOURCE_CLASS_OVERRIDES:
        lines.append("")
        lines.append("YOLO source-specific remapping overrides:")
        lines.append("")
        for source, overrides in sorted(SOURCE_CLASS_OVERRIDES.items()):
            for source_name, target_id in sorted(overrides.items()):
                target = "skip" if target_id is None else f"{target_id}:{TARGET_NAMES[target_id]}"
                lines.append(f"- `{source}` `{source_name}` -> {target}")
    lines.append("")
    lines.append("## LaneNet Summary")
    lines.append("")
    lines.append(f"- accepted images: {len(lane_samples)}")
    lines.append(f"- split counts: {format_counter(Counter(lane_split_counts))}")
    lines.append(
        f"- mask generation: LabelMe line/linestrip annotations rasterized with width {line_width}px; polygon annotations filled"
    )
    lines.append(f"- total lane shape instances: {sum(lane_shapes.values())}")
    lines.append("")
    lines.append("| source | seen labels | accepted images | skipped | shape labels |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for source in sorted(set(stats.source_seen) | set(stats.source_accepted)):
        if source.startswith(("g01_lanenet", "g23_roadline", "liu_images", "g04_lanenet")):
            lines.append(
                f"| {source} | {stats.source_seen[source]} | {stats.source_accepted[source]} | "
                f"{format_counter(stats.source_skipped[source])} | "
                f"{format_counter(stats.lane_line_labels[source])} |"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in stats.notes:
        lines.append(f"- {note}")
    lines.append(
        "- Stop sign labels were left out because the current project YOLO config has no `stop` class."
    )
    lines.append(
        "- If the task later adds a stop-sign behavior, add a class to the config and rerun this script with an updated mapping."
    )
    lines.append(
        "- Generated datasets under data/yolo and data/lanenet_hnet are ignored by git; keep the script and this report as the reproducible source of truth."
    )
    lines.append("")
    write_text(report_path, "\n".join(lines))


def main() -> None:
    args = parse_args()
    ensure_ratios(args.train, args.val, args.test)
    if not args.import_root.exists():
        raise FileNotFoundError(args.import_root)
    if args.lane_line_width < 1:
        raise ValueError("--lane-line-width must be positive")

    if not args.no_clean:
        clean_outputs(args.output_yolo, args.output_lanenet)

    yolo_samples, lane_samples, stats = collect_all(args.import_root)
    park_relabel_dirs = args.park_relabel_dir or list(DEFAULT_PARK_RELABEL_DIRS)
    yolo_samples = apply_corrected_park_labels(yolo_samples, stats, park_relabel_dirs)
    if not yolo_samples:
        raise RuntimeError("No YOLO samples were imported.")
    if not lane_samples:
        raise RuntimeError("No LaneNet samples were imported.")

    yolo_split_counts = write_yolo_dataset(
        yolo_samples,
        args.output_yolo,
        args.seed,
        args.train,
        args.val,
    )
    lane_split_counts = write_lanenet_dataset(
        lane_samples,
        args.output_lanenet,
        args.seed,
        args.train,
        args.val,
        args.lane_line_width,
    )
    write_report(
        args.report,
        stats,
        yolo_samples,
        lane_samples,
        yolo_split_counts,
        lane_split_counts,
        args.lane_line_width,
    )

    print(f"YOLO samples: {len(yolo_samples)} -> {args.output_yolo}")
    print(f"LaneNet samples: {len(lane_samples)} -> {args.output_lanenet}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
