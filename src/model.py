"""
A small, honest convolutional classifier for bird-call mel spectrograms.

This is the open-weight core of the project: we train it ourselves on openly
licensed field recordings (NPS Rocky Mountain sound library) and export it to
ONNX so the inference path is independent of any one framework and runs fully
offline.

It is intentionally small and limited. It discriminates among 5 species it was
trained on. It is not a general bird-identification service. That limitation is
the point: the open, swappable design is what makes it useful to extend with
your own local birds. See README for the model card.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BirdCallNet(nn.Module):
    """Small CNN over a (n_mels x n_frames) mel spectrogram."""

    def __init__(
        self,
        n_mels: int = 64,
        n_frames: int = 259,
        n_classes: int = 5,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3,
        embed_size: int = 64,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_classes = n_classes

        in_ch = 1
        layers = []
        for c in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, c, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size),
            ])
            in_ch = c
        self.conv = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(in_ch * 4 * 4, embed_size)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_mels, n_frames) -> (batch, 1, n_mels, n_frames)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Width/depth variants of BirdCallNet. "tiny" is the original shipped model; the
# others exist so a bigger dataset can feed a bigger net. export.py, identify.py
# and train.py all read this table so a checkpoint always reloads into the
# architecture it was trained with.
ARCHS: dict[str, dict] = {
    "tiny":  {"conv_channels": (16, 32, 64), "embed_size": 64},
    "small": {"conv_channels": (32, 64, 128), "embed_size": 128},
    "base":  {"conv_channels": (32, 64, 128, 256), "embed_size": 256},
    "large": {"conv_channels": (48, 96, 192, 384), "embed_size": 512},
}


def arch_kwargs(arch: str | None) -> dict:
    """Constructor kwargs for a named variant (unknown/None -> the shipped tiny net)."""
    return dict(ARCHS.get(arch or "tiny", {}))


def build_model(cfg: dict, n_classes: int | None = None) -> BirdCallNet:
    n_classes = n_classes if n_classes is not None else len(cfg["species"])
    return BirdCallNet(
        n_mels=cfg["n_mels"],
        n_frames=_expected_frames(cfg),
        n_classes=n_classes,
        **arch_kwargs(cfg.get("arch")),
    )


def _expected_frames(cfg: dict) -> int:
    sr = cfg["sr"]
    win_sec = cfg["window_sec"]
    hop = cfg["hop_length"]
    return int(win_sec * sr / hop) + 1


if __name__ == "__main__":
    # Quick shape/parameter sanity check.
    import json
    from pathlib import Path

    root = Path("model.py").resolve().parent
    cfg = json.loads((root / "data" / "dataset" / "config.json").read_text())
    m = build_model(cfg)
    print("model:", m)
    print("trainable params:", count_params(m))
    dummy = torch.randn(2, cfg["n_mels"], _expected_frames(cfg))
    out = m(dummy)
    print("input shape:", dummy.shape, "-> output shape:", out.shape)
