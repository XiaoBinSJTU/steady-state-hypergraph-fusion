"""Loss functions used by supervised anomaly-monitoring experiments."""

from __future__ import annotations

import torch
from torch import nn


class BinaryMonitoringLoss(nn.Module):
    def __init__(self, positive_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("positive_weight", torch.tensor([positive_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1), targets.float().reshape(-1), pos_weight=self.positive_weight
        )

