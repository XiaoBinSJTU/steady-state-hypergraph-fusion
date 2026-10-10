"""Deterministic synthetic inputs for demonstration only."""

from __future__ import annotations

import math

import torch

from .features import make_node_attributes
from .hypergraph import build_physical_hypergraph


DEMO_MODEL_CONFIG = {
    "spatial_dimension": 16,
    "propagation_layers": 2,
    "attention_heads": 4,
    "short_dilations": [1, 2],
    "long_dilations": [2, 4],
    "dropout": 0.0,
}


def make_demo_batch(
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Create a tiny non-industrial batch that exercises the complete forward path."""
    node_types = ["vibration", "acoustic", "current", "temperature", "virtual_band", "vibration"]
    positions = [0.0, 0.18, 0.4, 0.62, 0.8, 1.0]
    attributes, type_ids = make_node_attributes(node_types, positions)
    graph = build_physical_hypergraph(node_types)

    steps = 24
    time = torch.linspace(0.0, 2.0 * math.pi, steps)
    phases = torch.linspace(0.0, math.pi / 2.0, len(node_types))
    normal = torch.sin(time[:, None] + phases[None, :])
    coordinated_shift = 0.15 * torch.cos(2.0 * time[:, None] + phases[None, :])
    signals = torch.stack([normal, normal + coordinated_shift], dim=0).float().to(device)

    static = {
        "node_attributes": torch.from_numpy(attributes).to(device),
        "incidence": torch.from_numpy(graph.incidence).to(device),
        "hyperedge_weights": torch.from_numpy(graph.weights).to(device),
        "node_type_ids": torch.from_numpy(type_ids).to(device),
    }
    targets = torch.tensor([0.0, 1.0], device=device)
    return signals, static, targets
