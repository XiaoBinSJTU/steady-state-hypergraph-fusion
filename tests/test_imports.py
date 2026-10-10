"""Import checks for the public package."""

from steady_state_hypergraph_fusion import BinaryMonitoringLoss, HypergraphPropagation, ProposedModel


def test_public_imports() -> None:
    assert ProposedModel is not None
    assert HypergraphPropagation is not None
    assert BinaryMonitoringLoss is not None
