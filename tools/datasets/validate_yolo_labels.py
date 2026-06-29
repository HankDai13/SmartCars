#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO txt labels.")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--num-classes", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []

    for path in sorted(args.labels.rglob("*.txt")):
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) != 5:
                errors.append(f"{path}:{line_no}: expected 5 fields, got {len(parts)}")
                continue
            try:
                class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(f"{path}:{line_no}: non-numeric field")
                continue
            if not 0 <= class_id < args.num_classes:
                errors.append(f"{path}:{line_no}: class_id {class_id} out of range")
            if any(v < 0.0 or v > 1.0 for v in values):
                errors.append(f"{path}:{line_no}: bbox values must be normalized to [0, 1]")

    if errors:
        print("\n".join(errors[:100]))
        raise SystemExit(f"Found {len(errors)} label errors.")

    print("YOLO labels are valid.")

