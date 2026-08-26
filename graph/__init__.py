"""
graph/
======
Graph branch public interface for the Dynamic Bipartite Temporal GNN pipeline.

This package exposes the canonical output interface for GNN predictions.
Do NOT import fusion logic from here — the global fusion engine is Person 2's responsibility.
"""

from graph.gnn_output import GNNOutputRecord
from graph.inference_modes import InferenceMode
from graph.bipartite_graph_builder import BipartiteGraphBuilder

__all__ = ["GNNOutputRecord", "InferenceMode", "BipartiteGraphBuilder"]
