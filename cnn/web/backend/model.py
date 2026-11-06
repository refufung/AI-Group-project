"""Model architectures for inference (must match training checkpoints).

SmallCNN  — original custom-CNN track (cnn/outputs/best_small_cnn.pt)
ResNetSE  — new from-scratch champion (resBuildScratch track, arch="resnet_se")

Use build_model() to construct the right class from a checkpoint's "arch" key.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        channels = [64, 128, 256, 512]
        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class SE(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class BasicBlockSE(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, se_reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SE(out_ch, se_reduction)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity, inplace=True)


class ResNetSE(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.3,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        layers: tuple[int, ...] = (2, 2, 2, 2),
        se_reduction: int = 16,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        in_ch = channels[0]
        for stage_idx, (out_ch, n_blocks) in enumerate(zip(channels, layers)):
            for block_idx in range(n_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                stages.append(BasicBlockSE(in_ch, out_ch, stride=stride, se_reduction=se_reduction))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


ARCHS = {
    "small_cnn": SmallCNN,
    "resnet_se": ResNetSE,
}


def build_model(arch: str, num_classes: int, dropout: float) -> nn.Module:
    """Construct a model from a checkpoint's arch key (default: small_cnn)."""
    cls = ARCHS.get(arch, SmallCNN)
    return cls(num_classes=num_classes, dropout=dropout)
