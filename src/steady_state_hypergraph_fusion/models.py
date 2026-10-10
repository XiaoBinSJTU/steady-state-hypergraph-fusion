"""Core steady-state spatial guided spatiotemporal hypergraph model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .hypergraph import HypergraphPropagation


@dataclass(frozen=True)
class AblationOptions:
    use_ssg: bool = True
    use_haa: bool = True
    use_bp: bool = True
    use_rc: bool = True
    use_tp: bool = True


class CausalTemporalEncoder(nn.Module):
    def __init__(self, dimension: int, dilations: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_channels = 1
        for dilation in dilations:
            layers.append(
                nn.Conv1d(
                    input_channels,
                    dimension,
                    kernel_size=3,
                    dilation=dilation,
                    padding=2 * dilation,
                )
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_channels = dimension
        self.layers = nn.ModuleList(layers)
        self.dilations = dilations

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes = signals.shape
        encoded = signals.permute(0, 2, 1).reshape(batch * nodes, 1, steps)
        convolution_index = 0
        for layer in self.layers:
            encoded = layer(encoded)
            if isinstance(layer, nn.Conv1d):
                crop = 2 * self.dilations[convolution_index]
                encoded = encoded[..., :-crop]
                convolution_index += 1
        return encoded[..., -1].reshape(batch, nodes, -1)


class SteadyStateSpatialEncoder(nn.Module):
    def __init__(
        self,
        attribute_dimension: int,
        dimension: int,
        layers: int,
        dropout: float,
        options: AblationOptions,
    ) -> None:
        super().__init__()
        self.options = options
        self.signal_projection = nn.Linear(1, dimension)
        self.attribute_projection = nn.Linear(attribute_dimension, dimension)
        self.coherence_projection = nn.Sequential(nn.Linear(1, dimension), nn.Tanh())
        self.propagation = nn.ModuleList(
            [HypergraphPropagation(dimension, options.use_haa, options.use_bp) for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, signals: torch.Tensor, static: dict[str, torch.Tensor]) -> torch.Tensor:
        node_attributes = static["node_attributes"]
        node_features = self.signal_projection(signals.unsqueeze(-1))
        node_features = node_features + self.attribute_projection(node_attributes)[None, None, :, :]
        for propagation in self.propagation:
            update = propagation(
                node_features,
                static["incidence"],
                static["hyperedge_weights"],
                static["node_type_ids"],
            )
            node_features = self.dropout(node_features + update)
        if self.options.use_tp:
            pooled = node_features.mean(dim=1)
        else:
            pooled = node_features[:, -1]
        if self.options.use_bp:
            incidence = static["incidence"]
            edge_degree = incidence.sum(dim=0).clamp_min(1.0)
            edge_mean = torch.einsum("ne,btn->bte", incidence, signals)
            edge_mean = edge_mean / edge_degree[None, None, :]
            edge_power = torch.einsum("ne,btn->bte", incidence, signals.square())
            edge_power = edge_power / edge_degree[None, None, :]
            if self.options.use_tp:
                coherence = edge_mean.square().mean(dim=1) / edge_power.mean(dim=1).clamp_min(1e-6)
            else:
                coherence = edge_mean[:, -1].square() / edge_power[:, -1].clamp_min(1e-6)
            node_degree = incidence.sum(dim=1).clamp_min(1.0)
            node_coherence = torch.einsum("ne,be->bn", incidence, coherence)
            node_coherence = node_coherence / node_degree[None, :]
            pooled = pooled + self.coherence_projection(node_coherence.unsqueeze(-1))
        return pooled


class ProposedModel(nn.Module):
    def __init__(self, attribute_dimension: int, model_config: dict[str, Any], options: AblationOptions) -> None:
        super().__init__()
        dimension = int(model_config["spatial_dimension"])
        dropout = float(model_config["dropout"])
        heads = int(model_config["attention_heads"])
        self.options = options
        self.spatial_encoder = SteadyStateSpatialEncoder(
            attribute_dimension,
            dimension,
            int(model_config["propagation_layers"]),
            dropout,
            options,
        )
        self.short_encoder = CausalTemporalEncoder(dimension, list(model_config["short_dilations"]), dropout)
        self.long_encoder = CausalTemporalEncoder(dimension, list(model_config["long_dilations"]), dropout)
        self.norm_spatial = nn.LayerNorm(dimension)
        self.norm_short = nn.LayerNorm(dimension)
        self.norm_long = nn.LayerNorm(dimension)
        self.short_screen = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.long_screen = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.short_correction = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.long_correction = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.coordination_residual_scale = nn.Parameter(torch.tensor(2.0))
        self.classifier = nn.Sequential(
            nn.Linear(dimension * 2 + 1, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )

    def forward(self, signals: torch.Tensor, static: dict[str, torch.Tensor]) -> torch.Tensor:
        short = self.norm_short(self.short_encoder(signals))
        long = self.norm_long(self.long_encoder(signals))
        if self.options.use_ssg:
            spatial = self.norm_spatial(self.spatial_encoder(signals, static))
        else:
            spatial = self.norm_spatial(0.5 * (short + long))
        short_screened, _ = self.short_screen(spatial, short, short, need_weights=False)
        long_screened, _ = self.long_screen(spatial, long, long, need_weights=False)
        if self.options.use_rc:
            short_correction, _ = self.short_correction(short_screened, spatial, spatial, need_weights=False)
            long_correction, _ = self.long_correction(long_screened, spatial, spatial, need_weights=False)
            short_screened = short_screened + short_correction + 0.25 * spatial
            long_screened = long_screened + long_correction + 0.25 * spatial
        if self.options.use_ssg and self.options.use_bp:
            incidence = static["incidence"]
            edge_degree = incidence.sum(dim=0).clamp_min(1.0)
            edge_mean = torch.einsum("ne,btn->bte", incidence, signals)
            edge_mean = edge_mean / edge_degree[None, None, :]
            edge_power = torch.einsum("ne,btn->bte", incidence, signals.square())
            edge_power = edge_power / edge_degree[None, None, :]
            if self.options.use_tp:
                coordination = edge_mean.square().mean(dim=1) / edge_power.mean(dim=1).clamp_min(1e-6)
            else:
                coordination = edge_mean[:, -1].square() / edge_power[:, -1].clamp_min(1e-6)
            coordination_score = coordination.mean(dim=1, keepdim=True)
        else:
            coordination_score = torch.zeros(
                signals.shape[0], 1, dtype=signals.dtype, device=signals.device
            )
        fused = torch.cat(
            [short_screened.mean(dim=1), long_screened.mean(dim=1), coordination_score], dim=-1
        )
        logits = self.classifier(fused).squeeze(-1)
        if self.options.use_ssg and self.options.use_rc:
            logits = logits + torch.nn.functional.softplus(self.coordination_residual_scale) * (
                coordination_score.squeeze(-1) - 0.5
            )
        return logits


def options_for_variant(variant: str) -> AblationOptions:
    mapping = {
        "Full": AblationOptions(),
        "w/o SSG": AblationOptions(use_ssg=False),
        "w/o HAA": AblationOptions(use_haa=False),
        "w/o BP": AblationOptions(use_bp=False),
        "w/o RC": AblationOptions(use_rc=False),
        "w/o TP": AblationOptions(use_tp=False),
    }
    if variant not in mapping:
        raise KeyError(f"Unknown ablation variant: {variant}")
    return mapping[variant]

