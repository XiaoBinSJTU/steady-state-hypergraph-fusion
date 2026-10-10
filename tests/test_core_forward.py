"""Forward-path checks using deterministic synthetic inputs."""

import torch

from steady_state_hypergraph_fusion.demo import DEMO_MODEL_CONFIG, make_demo_batch
from steady_state_hypergraph_fusion.models import AblationOptions, ProposedModel


def test_core_forward_is_finite() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    signals, static, targets = make_demo_batch(device)
    model = ProposedModel(
        attribute_dimension=static["node_attributes"].shape[1],
        model_config=DEMO_MODEL_CONFIG,
        options=AblationOptions(),
    )
    logits = model(signals, static)
    assert logits.shape == targets.shape
    assert torch.isfinite(logits).all()


def test_ablation_switch_changes_executed_path() -> None:
    device = torch.device("cpu")
    signals, static, targets = make_demo_batch(device)
    full = ProposedModel(static["node_attributes"].shape[1], DEMO_MODEL_CONFIG, AblationOptions())
    without_spatial_guidance = ProposedModel(
        static["node_attributes"].shape[1],
        DEMO_MODEL_CONFIG,
        AblationOptions(use_ssg=False),
    )
    full_output = full(signals, static)
    ablated_output = without_spatial_guidance(signals, static)
    assert full_output.shape == targets.shape
    assert ablated_output.shape == targets.shape
    assert not torch.equal(full_output, ablated_output)
