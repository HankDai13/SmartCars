from __future__ import annotations

import torch
from torch import nn


class LFNet(nn.Module):
    def __init__(self, angle_min: float = 0.0, angle_max: float = 135.0) -> None:
        super().__init__()
        self.angle_min = angle_min
        self.angle_max = angle_max

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        unit = self.regressor(x)
        return unit * (self.angle_max - self.angle_min) + self.angle_min


def build_lfnet(config: dict | None = None) -> LFNet:
    config = config or {}
    return LFNet(
        angle_min=float(config.get("angle_min", 0.0)),
        angle_max=float(config.get("angle_max", 135.0)),
    )

