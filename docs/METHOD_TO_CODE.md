# Method-to-Code Map

This document maps the manuscript's core method stages to the public implementation. 

| Method stage | Public implementation | Role |
| --- | --- | --- |
| Heterogeneous sensor attributes | `src/steady_state_hypergraph_fusion/features.py` → `make_node_attributes` | Encodes sensor type, normalized position, orientation, measurement characteristics, and subsystem role. |
| Physical-prior hypergraph construction | `src/steady_state_hypergraph_fusion/hypergraph.py` → `build_physical_hypergraph` | Builds fixed mechanical, electrical, thermal, and local-neighborhood hyperedges. |
| Hyperedge-aware bipartite propagation | `src/steady_state_hypergraph_fusion/hypergraph.py` → `HypergraphPropagation.forward` | Performs node-to-hyperedge aggregation, type-aware attention, weighted redistribution, and residual feature updates. |
| Steady-state spatial guidance | `src/steady_state_hypergraph_fusion/models.py` → `SteadyStateSpatialEncoder` | Combines dynamic signals with static attributes, stacked hypergraph propagation, temporal pooling, and coherence features. |
| Short- and long-term temporal encoding | `src/steady_state_hypergraph_fusion/models.py` → `CausalTemporalEncoder` | Uses causal dilated convolutions at separate temporal scales. |
| Spatially guided screening and residual correction | `src/steady_state_hypergraph_fusion/models.py` → `ProposedModel.forward` | Applies cross-attention between spatial, short-term, and long-term representations before binary monitoring inference. |
| Monitoring objective | `src/steady_state_hypergraph_fusion/losses.py` → `BinaryMonitoringLoss` | Computes weighted binary cross-entropy on model logits. |
| Mechanism switches | `src/steady_state_hypergraph_fusion/models.py` → `AblationOptions`, `options_for_variant` | Exposes code paths for inspecting individual components; this is not a packaged reproduction of the paper's ablation experiments. |

