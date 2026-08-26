"""
fusion/
=======
Multi-signal fusion engine for the Hybrid NIDS pipeline.

Combines XGBoost, Autoencoder, GNN, and Temporal correlation scores
into a single risk score using fixed architecture weights.
"""

from fusion.fusion_engine import (
    FusionEngine,
    load_fusion_threshold,
    _DEFAULT_THRESHOLD,
)

__all__ = [
    "FusionEngine",
    "load_fusion_threshold",
    "_DEFAULT_THRESHOLD",
]
