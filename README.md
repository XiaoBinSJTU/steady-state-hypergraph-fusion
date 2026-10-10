# Steady-State Spatial Guided Spatiotemporal Hypergraph Data Fusion for Digital Twin Monitoring of Industrial Equipment under Harsh Operating Conditions

[Bin Xiao on GitHub](https://github.com/XiaoBinSJTU)

## Paper Information

**Status:** Manuscript Under Review  
**Authors, venue, year, and DOI:** `TO_VERIFY`

This repository provides the core algorithm implementation and a research reference interface. 

## Motivation

![Motivation: steady-state monitoring challenges](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/05-steady-state-mssp-challenges.svg)

## Overall Framework

![Overall framework: steady-state spatial guided spatiotemporal hypergraph fusion](https://raw.githubusercontent.com/XiaoBinSJTU/XiaoBinSJTU/main/assets/graphical-abstracts/05-steady-state-mssp-framework.svg)

## Research Highlights

Harsh operating conditions can disrupt individual sensing channels and weaken purely temporal monitoring models. The implementation combines physical-prior hypergraph relations, heterogeneous sensor attributes, multi-scale causal temporal encoders, and spatially guided cross-attention. A coherence-aware residual path preserves coordinated responses across sensor groups while retaining an end-to-end differentiable monitoring objective.

## Method Overview

### 1. Physical-Prior Hypergraph

Sensor roles and normalized locations define mechanical, electrical, thermal, and local-neighborhood hyperedges. The public constructor returns an incidence matrix and fixed positive relation weights without reading equipment data.

### 2. Steady-State Spatial Encoding

Signal projections are combined with static node attributes. Stacked bipartite propagation aggregates node semantics into hyperedges and redistributes weighted context to member nodes.

### 3. Multi-Scale Temporal Encoding

Separate causal dilated-convolution branches summarize short- and long-term signal behavior for every sensor node.

### 4. Guided Fusion and Residual Correction

Spatial representations screen both temporal branches through cross-attention. A second correction stage and a hyperedge-coherence residual support the final monitoring logit.

### 5. Monitoring Objective

The model is trained through weighted binary cross-entropy. The public demo performs a real loss computation and backward pass but does not train on or evaluate industrial data.

## Method-to-Code Map

| Method component | Code entry |
| --- | --- |
| Sensor attributes | [`features.py`](src/steady_state_hypergraph_fusion/features.py) — `make_node_attributes` |
| Physical hypergraph | [`hypergraph.py`](src/steady_state_hypergraph_fusion/hypergraph.py) — `build_physical_hypergraph` |
| Bipartite propagation | [`hypergraph.py`](src/steady_state_hypergraph_fusion/hypergraph.py) — `HypergraphPropagation` |
| Spatial and temporal encoders | [`models.py`](src/steady_state_hypergraph_fusion/models.py) — `SteadyStateSpatialEncoder`, `CausalTemporalEncoder` |
| Guided fusion and inference | [`models.py`](src/steady_state_hypergraph_fusion/models.py) — `ProposedModel` |
| Monitoring loss | [`losses.py`](src/steady_state_hypergraph_fusion/losses.py) — `BinaryMonitoringLoss` |

See [`docs/METHOD_TO_CODE.md`](docs/METHOD_TO_CODE.md) for a more detailed mapping.

## Quick Start

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python examples\minimal_demo.py
python -m pytest -q
```

## Author

[XiaoBinSJTU](https://github.com/XiaoBinSJTU)
