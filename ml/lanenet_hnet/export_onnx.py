#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import build_lanenet_hnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaneNet + H-Net checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height", default=256, type=int)
    parser.add_argument("--width", default=512, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_lanenet_hnet(checkpoint.get("model_cfg", {}))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dummy = torch.zeros(1, 3, args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["binary_logits", "embeddings", "h_params"],
        opset_version=12,
        dynamic_axes=None,
    )
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()

