"""Paper-guided static node attributes for heterogeneous sensor roles."""

from __future__ import annotations

import numpy as np


TYPE_NAMES = ("vibration", "acoustic", "current", "temperature", "virtual_band")


def make_node_attributes(node_types: list[str], positions: list[float]) -> tuple[np.ndarray, np.ndarray]:
    type_ids = np.asarray([TYPE_NAMES.index(name) for name in node_types], dtype=np.int64)
    one_hot = np.eye(len(TYPE_NAMES), dtype=np.float32)[type_ids]
    position_array = np.asarray(positions, dtype=np.float32)[:, None]
    orientation = np.stack(
        [np.sin(np.pi * position_array[:, 0]), np.cos(np.pi * position_array[:, 0])], axis=1
    ).astype(np.float32)
    measurement = np.zeros((len(node_types), 2), dtype=np.float32)
    subsystem = np.zeros((len(node_types), 3), dtype=np.float32)
    for index, name in enumerate(node_types):
        measurement[index] = {
            "vibration": (1.0, 1.0),
            "acoustic": (0.9, 1.0),
            "current": (0.6, 0.8),
            "temperature": (0.2, 0.3),
            "virtual_band": (0.8, 0.9),
        }[name]
        subsystem[index] = {
            "vibration": (1.0, 0.0, 0.0),
            "acoustic": (1.0, 0.0, 0.0),
            "current": (0.0, 1.0, 0.0),
            "temperature": (0.0, 0.0, 1.0),
            "virtual_band": (1.0, 0.0, 0.0),
        }[name]
    attributes = np.concatenate([position_array, orientation, measurement, one_hot, subsystem], axis=1)
    return attributes.astype(np.float32), type_ids

