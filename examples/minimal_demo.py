"""Run a minimal forward, loss, and backward pass on synthetic inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from steady_state_hypergraph_fusion.demo import DEMO_MODEL_CONFIG, make_demo_batch
from steady_state_hypergraph_fusion.losses import BinaryMonitoringLoss
from steady_state_hypergraph_fusion.models import AblationOptions, ProposedModel


def main() -> None:
    """Exercise the public core without datasets, checkpoints, or stored results."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signals, static, targets = make_demo_batch(device)
    model = ProposedModel(
        attribute_dimension=static["node_attributes"].shape[1],
        model_config=DEMO_MODEL_CONFIG,
        options=AblationOptions(),
    ).to(device)
    loss_function = BinaryMonitoringLoss().to(device)

    model.train()
    logits = model(signals, static)
    loss = loss_function(logits, targets)
    loss.backward()

    if logits.shape != targets.shape:
        raise RuntimeError(f"Unexpected output shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("The core computation produced a non-finite value.")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Backward validation failed.")

    device_mode = "GPU" if device.type == "cuda" else "CPU"
    print(f"Device mode: {device_mode}")
    print("import/forward/loss/backward passed")
    print("Synthetic inputs are for demonstration only; no paper metrics were computed.")


if __name__ == "__main__":
    main()
