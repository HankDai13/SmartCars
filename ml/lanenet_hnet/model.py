from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class LaneNetHNet(nn.Module):
    def __init__(self, embedding_dim: int = 4) -> None:
        super().__init__()
        self.enc1 = conv_block(3, 32, stride=2)
        self.enc2 = conv_block(32, 64, stride=2)
        self.bottleneck = conv_block(64, 128, stride=1)

        self.dec2 = conv_block(128 + 64, 64)
        self.dec1 = conv_block(64 + 32, 32)

        self.binary_head = nn.Conv2d(32, 1, kernel_size=1)
        self.embedding_head = nn.Conv2d(32, embedding_dim, kernel_size=1)

        self.hnet = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d0 = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)

        binary_logits = self.binary_head(d0)
        embeddings = self.embedding_head(d0)
        h_params = self.hnet(b)
        return binary_logits, embeddings, h_params


def build_lanenet_hnet(config: dict | None = None) -> LaneNetHNet:
    config = config or {}
    return LaneNetHNet(embedding_dim=int(config.get("embedding_dim", 4)))

