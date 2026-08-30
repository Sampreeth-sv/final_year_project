"""
fusion/
=======
Dynamic Multi-Model Fusion Engine for the Hybrid NIDS Pipeline.

Combines XGBoost, Autoencoder, GNN, and Temporal correlation scores
into a single risk score using dynamically calculated weights based on
entropy-based confidence and consistency-weighted reliability.
"""

from fusion.fusion_engine import (
    FusionEngine,
    calculate_entropy,
    calculate_normalized_entropy,
    calculate_confidence,
    calculate_consistency,
    calculate_reliability,
    normalize_temporal_score,
)

__all__ = [
    "FusionEngine",
    "calculate_entropy",
    "calculate_normalized_entropy",
    "calculate_confidence",
    "calculate_consistency",
    "calculate_reliability",
    "normalize_temporal_score",
]