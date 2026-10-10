"""Loss and hypergraph logic checks."""

import torch

from steady_state_hypergraph_fusion.demo import DEMO_MODEL_CONFIG, make_demo_batch
from steady_state_hypergraph_fusion.hypergraph import build_physical_hypergraph
from steady_state_hypergraph_fusion.losses import BinaryMonitoringLoss
from steady_state_hypergraph_fusion.models import AblationOptions, ProposedModel


def test_physical_hypergraph_has_valid_membership() -> None:
    graph = build_physical_hypergraph(["vibration", "acoustic", "current", "temperature"])
    assert graph.incidence.shape[0] == 4
    assert graph.incidence.shape[1] == graph.weights.shape[0]
    assert (graph.incidence.sum(axis=1) >= 1).all()
    assert (graph.weights > 0).all()


def test_loss_and_backward_are_finite() -> None:
    torch.manual_seed(11)
    device = torch.device("cpu")
    signals, static, targets = make_demo_batch(device)
    model = ProposedModel(static["node_attributes"].shape[1], DEMO_MODEL_CONFIG, AblationOptions())
    loss = BinaryMonitoringLoss()(model(signals, static), targets)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert torch.isfinite(loss)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
