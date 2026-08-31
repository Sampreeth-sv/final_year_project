"""
graph/gnn_output.py
===================
Canonical GNN Output Interface for the Dynamic Bipartite Temporal GNN Branch.

This module defines the SINGLE authoritative output format for GNN predictions.
Person 2's fusion engine should consume GNNOutputRecord instances alongside
xgb_prob and ae_score from LiveTrafficEngine to build the global fusion.

INTEGRATION CONTRACT:
  The GNN branch does NOT implement global fusion.
  It exposes only GNNOutputRecord, which Person 2 combines with upstream scores.

Example usage in Person 2's fusion engine:
    from graph.gnn_output import GNNOutputRecord

    # After running OnlineGraphStream:
    gnn_out = GNNOutputRecord(
        flow_id=flow_id,
        gnn_score=float(probs[-1]),
        gnn_prediction=int(probs[-1] >= 0.5),
        temporal_score=temporal_shift,
        graph_context={
            "host_id":     src_host,
            "service_id":  svc_id,
            "snapshot_window": stream.buffer_size,
        },
        gnn_latency_ms=gnn_latency_ms,
    )

    # Then fuse with upstream scores:
    final_score = person2_fusion(gnn_out.gnn_score, flow_rec["xgb_prob"], flow_rec["ae_score"])
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class GNNOutputRecord:
    """
    Canonical output record from the Dynamic Bipartite Temporal GNN.

    Fields:
        flow_id         : str  — unique identifier for the flow (e.g., '192.168.1.1:54321->80/6@t')
        gnn_score       : float [0.0, 1.0] — edge risk probability from GNN sigmoid output
        gnn_prediction  : int   {0, 1} — binary prediction (1 = attack) at threshold 0.5
        temporal_score  : float — GRU hidden state L2 norm, proxy for temporal activity level
        graph_context   : dict  — metadata: host_id, service_id, snapshot_window, mode
        gnn_latency_ms  : float — wall-clock GNN inference time in milliseconds

    NOTE: gnn_score is the ONLY GNN-produced value to be used in Person 2's fusion.
    gnn_prediction is a convenience field; Person 2 may apply a different threshold.
    temporal_score is supplementary context, not a classification output.
    """
    flow_id:        str
    gnn_score:      float
    gnn_prediction: int
    temporal_score: float
    graph_context:  Dict[str, Any] = field(default_factory=dict)
    gnn_latency_ms: float = 0.0

    def __post_init__(self):
        # Clamp gnn_score to valid range
        if not (0.0 <= self.gnn_score <= 1.0):
            raise ValueError(f"gnn_score must be in [0, 1], got {self.gnn_score}")
        if self.gnn_prediction not in (0, 1):
            raise ValueError(f"gnn_prediction must be 0 or 1, got {self.gnn_prediction}")
        if self.gnn_latency_ms < 0:
            raise ValueError(f"gnn_latency_ms must be >= 0, got {self.gnn_latency_ms}")
        # Validate temporal_score range - from GNN hidden state L2 norm
        # GRU hidden states can be arbitrarily large; we don't clamp here but document expectations

    def to_dict(self):
        """Serializable dict representation for logging and dashboard integration."""
        return asdict(self)

    @classmethod
    def from_stream_output(cls, flow_id, probs, gnn_latency_ms, stream, edge_idx=-1, temporal_shift=None):
        """
        Convenience constructor from OnlineGraphStream.evaluate_realtime_gnn_risk() output.

        Args:
            flow_id:        str identifier for this flow
            probs:          np.ndarray of edge probabilities (sigmoid of logits)
            gnn_latency_ms: float GNN inference time in ms
            stream:         OnlineGraphStream instance (for context)
            edge_idx:       int index into probs to extract (default: -1 = last edge)
            temporal_shift:  float L2 norm of GRU hidden states (supplementary context)

        Returns:
            GNNOutputRecord

        Output semantics:
        - gnn_score: sigmoid edge risk probability [0,1] from GNN classifier
        - temporal_score: L2 norm of GRU hidden states (supplementary context)
        - temporal_score is NOT a classification signal; it reflects GRU activation magnitude
        """
        import numpy as np
        if probs is None or len(probs) == 0:
            return cls(
                flow_id=flow_id,
                gnn_score=0.0,
                gnn_prediction=0,
                temporal_score=0.0,
                graph_context={"note": "empty_graph"},
                gnn_latency_ms=gnn_latency_ms,
            )

        score = float(np.clip(probs[edge_idx], 0.0, 1.0))

        # temporal_shift is now computed in evaluate_realtime_gnn_risk()
        # Fallback to stream state if not provided (for backward compatibility)
        if temporal_shift is None:
            temporal_shift = 0.0
            if stream.prev_host_h is not None:
                import torch
                temporal_shift = float(torch.norm(stream.prev_host_h).item())

        return cls(
            flow_id=flow_id,
            gnn_score=score,
            gnn_prediction=int(score >= 0.5),
            temporal_score=round(temporal_shift, 6),
            graph_context={
                "mode":             stream.mode,
                "snapshot_window": stream.buffer_size,
                "edge_idx":        int(edge_idx % len(probs)),
            },
            gnn_latency_ms=round(gnn_latency_ms, 3),
        )
