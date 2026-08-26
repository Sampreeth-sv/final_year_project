"""
gnn/
====
Graph Neural Network branch for the Hybrid NIDS pipeline.

Exposes the Dynamic Bipartite Temporal GNN model and the online
streaming wrapper that maintains a sliding-window bipartite graph
and produces per-edge risk scores in real time.
"""

from gnn.dynamic_temporal_gnn import (
    HeteroBipartiteConv,
    TemporalNodeUpdate,
    EdgeRiskClassifier,
    DynamicBipartiteTemporalGNN,
)
from gnn.online_graph_stream import OnlineGraphStream

__all__ = [
    "HeteroBipartiteConv",
    "TemporalNodeUpdate",
    "EdgeRiskClassifier",
    "DynamicBipartiteTemporalGNN",
    "OnlineGraphStream",
]
