"""From-scratch EfficientNet-B0 for 4-class dog breed classification.

No torchvision / ImageNet pretrained weights.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


def _round_channels(ch: int, width_mult: float, divisor: int = 8) -> int:
    ch = max(divisor, int(ch * width_mult + divisor / 2) // divisor * divisor)
    return ch


def _round_repeats(n: int, depth_mult: float) -> int:
    if abs(depth_mult - 1.0) < 1e-9:
        return max(1, int(n))
    return max(1, int(math.ceil(n * depth_mult)))


class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25):
        super().__init__()
        squeezed = max(1, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, squeezed, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeezed, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class MBConv(nn.Module):
    """Mobile inverted bottleneck + squeeze-excitation + optional DropPath."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        expand_ratio: int,
        kernel_size: int,
        stride: int,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.use_skip = stride == 1 and in_ch == out_ch
        mid_ch = int(in_ch * expand_ratio)

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, kernel_size=1))
        layers.append(ConvBNAct(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, groups=mid_ch))
        layers.append(SqueezeExcitation(mid_ch, se_ratio=se_ratio))
        layers.append(nn.Conv2d(mid_ch, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0 and self.use_skip else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_skip:
            out = x + self.drop_path(out)
        return out


# EfficientNet-B0 stage table: (expand, out_ch, num_blocks, stride, kernel)
_B0_BLOCKS: Sequence[tuple[int, int, int, int, int]] = (
    (1, 16, 1, 1, 3),
    (6, 24, 2, 2, 3),
    (6, 40, 2, 2, 5),
    (6, 80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
)


class EfficientNetB0(nn.Module):
    """EfficientNet-B0: stem → MBConv stages → head → GAP → dropout → linear."""

    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        depth_drop: int = 0,
        depth_mult: float = 1.0,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.width_mult = width_mult
        self.depth_drop = max(0, int(depth_drop))
        self.depth_mult = float(depth_mult)
        self.se_ratio = float(se_ratio)
        self.dropout_p = dropout
        self.drop_path_rate = float(drop_path_rate)
        stem_ch = _round_channels(32, width_mult)
        head_ch = _round_channels(1280, width_mult)

        deep_stage_idx = 5
        block_specs: list[tuple[int, int, int, int, int]] = []
        for stage_i, (expand, out_ch, n_blocks, stride, k) in enumerate(_B0_BLOCKS):
            n = _round_repeats(n_blocks, self.depth_mult)
            if stage_i == deep_stage_idx and self.depth_drop > 0:
                n = max(1, n - self.depth_drop)
            for i in range(n):
                s = stride if i == 0 else 1
                block_specs.append((expand, out_ch, s, k, stage_i))

        n_blocks_total = max(len(block_specs), 1)
        features: list[nn.Module] = [ConvBNAct(3, stem_ch, kernel_size=3, stride=2)]
        in_ch = stem_ch
        for bi, (expand, out_ch, stride, k, _) in enumerate(block_specs):
            out_ch_r = _round_channels(out_ch, width_mult)
            dp = self.drop_path_rate * bi / (n_blocks_total - 1) if n_blocks_total > 1 else 0.0
            features.append(
                MBConv(
                    in_ch,
                    out_ch_r,
                    expand_ratio=expand,
                    kernel_size=k,
                    stride=stride,
                    se_ratio=self.se_ratio,
                    drop_path=dp,
                )
            )
            in_ch = out_ch_r
        features.append(ConvBNAct(in_ch, head_ch, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(head_ch, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_efficientnet_b0(
    num_classes: int = 4,
    pretrained: bool = False,
    width_mult: float = 0.5,
    dropout: float = 0.2,
    depth_drop: int = 0,
    depth_mult: float = 1.0,
    se_ratio: float = 0.25,
    drop_path_rate: float = 0.0,
) -> EfficientNetB0:
    """Build from-scratch EfficientNet-B0 (no ImageNet weights)."""
    if pretrained:
        raise ValueError(
            "From-scratch EfficientNet only; pass pretrained=False "
            "(torchvision ImageNet weights are not used)."
        )
    return EfficientNetB0(
        num_classes=num_classes,
        dropout=dropout,
        width_mult=width_mult,
        depth_drop=depth_drop,
        depth_mult=depth_mult,
        se_ratio=se_ratio,
        drop_path_rate=drop_path_rate,
    )


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable
