"""Fixed physical-prior hypergraph construction and propagation layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class HypergraphArrays:
    incidence: np.ndarray
    weights: np.ndarray


def build_physical_hypergraph(node_types: list[str]) -> HypergraphArrays:
    nodes = len(node_types)
    edges: list[list[int]] = []
    weights: list[float] = []

    mechanical = [i for i, name in enumerate(node_types) if name in {"vibration", "acoustic", "virtual_band"}]
    electrical = [i for i, name in enumerate(node_types) if name == "current"]
    thermal = [i for i, name in enumerate(node_types) if name == "temperature"]

    if mechanical:
        midpoint = max(1, len(mechanical) // 2)
        for group in (mechanical[: midpoint + 1], mechanical[midpoint:]):
            if len(group) >= 2:
                edges.append(sorted(set(group)))
                weights.append(1.0)
    if mechanical and electrical:
        edges.append(mechanical + electrical)
        weights.append(0.9)
    if electrical and thermal:
        edges.append(electrical + thermal)
        weights.append(0.75)
    if mechanical and thermal:
        edges.append([mechanical[0], thermal[0]])
        weights.append(0.55)
    for left in range(nodes - 1):
        edges.append([left, left + 1])
        delay = abs(left - (left + 1))
        weights.append(float(np.exp(-0.18 * delay)))
    if not edges:
        edges.append(list(range(nodes)))
        weights.append(1.0)

    incidence = np.zeros((nodes, len(edges)), dtype=np.float32)
    for edge_index, members in enumerate(edges):
        incidence[members, edge_index] = 1.0
    return HypergraphArrays(incidence=incidence, weights=np.asarray(weights, dtype=np.float32))


class HypergraphPropagation(nn.Module):
    def __init__(self, dimension: int, use_attention: bool = True, use_bipartite: bool = True) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.use_bipartite = use_bipartite
        self.attention = nn.Sequential(
            nn.Linear(dimension * 2 + 1, dimension),
            nn.GELU(),
            nn.Linear(dimension, 1),
        )
        self.update = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.GELU(), nn.LayerNorm(dimension))

    def forward(
        self,
        node_features: torch.Tensor,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        node_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_bipartite:
            return self.update(torch.cat([node_features, node_features], dim=-1))
        edge_degree = incidence.sum(dim=0).clamp_min(1.0)
        edge_semantics = torch.einsum("ne,btnd->bted", incidence, node_features)
        edge_semantics = edge_semantics / edge_degree[None, None, :, None]
        if self.use_attention:
            expanded_nodes = node_features[:, :, :, None, :].expand(-1, -1, -1, incidence.shape[1], -1)
            expanded_edges = edge_semantics[:, :, None, :, :].expand(-1, -1, incidence.shape[0], -1, -1)
            edge_type_mean = torch.einsum(
                "ne,n->e", incidence, node_type_ids.float()
            ) / edge_degree
            type_delta = (node_type_ids.float()[:, None] - edge_type_mean[None, :]).abs()
            delta = type_delta[None, None, :, :, None].expand(*expanded_nodes.shape[:-1], 1)
            logits = self.attention(torch.cat([expanded_nodes, expanded_edges, delta], dim=-1)).squeeze(-1)
            logits = logits.masked_fill(incidence[None, None, :, :] == 0, -1e4)
            alpha = torch.softmax(logits, dim=-1) * incidence[None, None, :, :]
        else:
            node_degree = incidence.sum(dim=1).clamp_min(1.0)
            alpha = incidence[None, None, :, :] / node_degree[None, None, :, None]
        edge_features = torch.einsum("btne,btnd->bted", alpha, node_features)
        edge_features = edge_features / incidence.sum(dim=0).clamp_min(1.0)[None, None, :, None]
        weighted = edge_features * edge_weights[None, None, :, None]
        redistributed = torch.einsum("ne,bted->btnd", incidence, weighted)
        normalizer = torch.einsum("ne,e->n", incidence, edge_weights).clamp_min(1e-6)
        redistributed = redistributed / normalizer[None, None, :, None]
        return self.update(torch.cat([node_features, redistributed], dim=-1))

