"""Core implementation of steady-state guided spatiotemporal hypergraph fusion."""

from .features import TYPE_NAMES, make_node_attributes
from .hypergraph import HypergraphArrays, HypergraphPropagation, build_physical_hypergraph
from .losses import BinaryMonitoringLoss
from .models import (
    AblationOptions,
    CausalTemporalEncoder,
    ProposedModel,
    SteadyStateSpatialEncoder,
    options_for_variant,
)

__all__ = [
    "AblationOptions",
    "BinaryMonitoringLoss",
    "CausalTemporalEncoder",
    "HypergraphArrays",
    "HypergraphPropagation",
    "ProposedModel",
    "SteadyStateSpatialEncoder",
    "TYPE_NAMES",
    "build_physical_hypergraph",
    "make_node_attributes",
    "options_for_variant",
]

__version__ = "0.1.0"
