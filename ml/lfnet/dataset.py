from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LFNetRecord:
    image: Path
    angle: float


def load_records(csv_path: Path, image_root: Path) -> list[LFNetRecord]:
    records: list[LFNetRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "image" not in reader.fieldnames or "angle" not in reader.fieldnames:
            raise ValueError("LFNet labels CSV must contain image and angle columns.")
        for row in reader:
            image_path = image_root / row["image"]
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            records.append(LFNetRecord(image=image_path, angle=float(row["angle"])))
    if not records:
        raise ValueError(f"No records loaded from {csv_path}")
    return records


def split_records(
    records: list[LFNetRecord], val_ratio: float, seed: int = 42
) -> tuple[list[LFNetRecord], list[LFNetRecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


class LFNetDataset(Dataset):
    def __init__(self, records: list[LFNetRecord], width: int, height: int) -> None:
        self.records = records
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = cv2.imread(str(record.image), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        angle = np.array([record.angle], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(angle)

