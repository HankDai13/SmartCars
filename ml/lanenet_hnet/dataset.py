from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LaneRecord:
    image: Path
    binary_mask: Path
    instance_mask: Path
    split: str


def load_records(
    split_csv: Path,
    image_root: Path,
    binary_mask_root: Path,
    instance_mask_root: Path,
) -> list[LaneRecord]:
    records: list[LaneRecord] = []
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError("splits.csv must contain image and split columns.")
        for row in reader:
            image = image_root / row["image"]
            stem = Path(row["image"]).stem
            binary = first_existing(binary_mask_root, stem)
            instance = first_existing(instance_mask_root, stem)
            for path in (image, binary, instance):
                if not path.exists():
                    raise FileNotFoundError(path)
            records.append(
                LaneRecord(
                    image=image,
                    binary_mask=binary,
                    instance_mask=instance,
                    split=row["split"].strip().lower(),
                )
            )
    if not records:
        raise ValueError(f"No records loaded from {split_csv}")
    return records


def first_existing(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    return root / f"{stem}.png"


class LaneNetDataset(Dataset):
    def __init__(self, records: list[LaneRecord], width: int, height: int) -> None:
        self.records = records
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = cv2.imread(str(record.image), cv2.IMREAD_COLOR)
        binary = cv2.imread(str(record.binary_mask), cv2.IMREAD_GRAYSCALE)
        instance = cv2.imread(str(record.instance_mask), cv2.IMREAD_GRAYSCALE)
        if image is None or binary is None or instance is None:
            raise FileNotFoundError(record)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        binary = cv2.resize(binary, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        instance = cv2.resize(instance, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        binary = (binary > 127).astype(np.float32)[None, ...]
        instance = instance.astype(np.int64)

        return (
            torch.from_numpy(np.transpose(image, (2, 0, 1))),
            torch.from_numpy(binary),
            torch.from_numpy(instance),
        )

