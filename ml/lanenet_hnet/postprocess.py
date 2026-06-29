from __future__ import annotations

import cv2
import numpy as np


def binary_lane_offset(binary_logits: np.ndarray, threshold: float = 0.5) -> tuple[float, float, bool]:
    """Return lateral offset estimate, confidence, valid flag from a binary lane map.

    The offset is normalized to [-1, 1], negative means lane center is left of image center.
    This is a lightweight fallback for the decision node; production code can replace it
    with embedding clustering and polynomial lane fitting.
    """

    logits = np.squeeze(binary_logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    mask = (probs > threshold).astype(np.uint8)
    height, width = mask.shape[-2:]
    lower_half = mask[height // 2 :, :]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lower_half, connectivity=8)
    if num_labels <= 1:
        return 0.0, 0.0, False

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    cx = float(centroids[best][0])
    confidence = float(min(1.0, areas[best - 1] / max(width * height * 0.05, 1.0)))
    offset = (cx - width / 2.0) / (width / 2.0)
    return offset, confidence, True

