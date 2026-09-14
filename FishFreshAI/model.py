"""EfficientNet-B0 based binary classifier for FishFresh AI."""

import torch
import torch.nn as nn
from typing import Any

from efficientnet_pytorch import EfficientNet


class FishFreshEfficientNet(nn.Module):
    """Binary classifier built on top of EfficientNet-B0 (ImageNet pretrained)."""

    def __init__(self, num_classes: int = 1, dropout: float = 0.3):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")

        # Optionally freeze backbone parameters (helps with overfitting)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the original classifier head.
        # EfficientNet's final layer is self._fc
        in_features = self.backbone._fc.in_features

        self.backbone._fc = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through backbone and head; return logits."""
        return self.backbone(x)