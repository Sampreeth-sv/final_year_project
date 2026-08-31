"""
modules/online_graph_stream.py
==============================
PHASE 7: Online Streaming Bipartite Graph Maintenance & GNN State Tracker

Maintains a SLIDING-WINDOW temporal bipartite graph (V_host, V_service, E) in memory.
The graph is REBUILT from the last `window_size` flows on each inference call to
prevent unbounded memory growth. Old flow statistics are NOT accumulated forever.

GRU state is propagated across inference calls for temporal continuity. When the
graph topology changes significantly (node set changes after a window rebuild),
GRU hidden states are re-initialized to avoid shape mismatches.

Modes:
  OFFLINE_REPLAY: Use with replayed/historical flow records (not "live").
  LIVE_CAPTURE  : Use with flows produced by FlowExtractor from real packet capture.

See graph/inference_modes.py for the InferenceMode enum.

Temporal Window:
  The stream maintains a bounded buffer of the last `window_size` flows.
  Default window_size=500 means the graph reflects the most recent 500 flows.
  Older flows are evicted from both the buffer AND the graph on each rebuild.

Thread Safety:
  ingest_flow() and evaluate_realtime_gnn_risk() are protected by a threading.Lock.
  They are safe to call from a background capture thread and a separate inference thread.
"""

import time
import threading
import logging
import torch
import numpy as np
from graph.bipartite_graph_builder import BipartiteGraphBuilder
from gnn.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN

logger = logging.getLogger(__name__)


class OnlineGraphStream:
    """
    Sliding-Window Online Bipartite Graph & GNN State Tracker.

    Args:
        gnn_model:   trained DynamicBipartiteTemporalGNN instance
        scalers:     dict with keys 'host_scaler', 'service_scaler', 'edge_scaler'
        window_size: maximum number of recent flows to maintain in the graph
        mode:        string label for logging — 'offline_replay' or 'live_capture'
    """

    def __init__(self, gnn_model, scalers, window_size=500, mode="live_capture"):
        self.gnn_model      = gnn_model
        self.host_scaler    = scalers["host_scaler"]
        self.service_scaler = scalers["service_scaler"]
        self.edge_scaler    = scalers["edge_scaler"]
        self.window_size    = window_size
        self.mode           = mode

        # Bounded circular flow buffer
        self.flow_buffer = []
        self._lock = threading.Lock()

        # GRU hidden states (None = zero init on first call)
        self.prev_host_h    = None
        self.prev_service_h = None

        # Track previous graph topology for GRU-state reset detection
        self._prev_num_hosts    = 0
        self._prev_num_services = 0

        logger.info(
            "OnlineGraphStream initialized: mode=%s, window_size=%d",
            mode, window_size,
        )

    def ingest_flow(self, flow_rec):
        """
        Ingests a single flow record into the sliding-window buffer.
        Thread-safe.

        Args:
            flow_rec: dict — output from LiveTrafficEngine.process_flow_record()
                      Must contain: src_ip, dst_ip, dst_port, protocol, timestamp,
                      byte_count, pkt_count, duration, xgb_prob, ae_score,
                      fusion_score. Optional: attack_label.
        """
        with self._lock:
            self.flow_buffer.append(flow_rec)
            # Enforce window size — evict oldest flows
            if len(self.flow_buffer) > self.window_size:
                self.flow_buffer = self.flow_buffer[-self.window_size:]

    def _rebuild_graph_from_buffer(self):
        """
        Rebuilds the BipartiteGraphBuilder from scratch using the current flow_buffer.
        This ensures old flows evicted from the buffer do not persist in node statistics.
        Called internally before each GNN inference.
        """
        builder = BipartiteGraphBuilder()
        for flow_rec in self.flow_buffer:
            builder.add_flow(
                src_ip=flow_rec["src_ip"],
                dst_ip=flow_rec["dst_ip"],
                dst_port=flow_rec["dst_port"],
                protocol=flow_rec["protocol"],
                timestamp=flow_rec["timestamp"],
                byte_count=flow_rec["byte_count"],
                pkt_count=flow_rec["pkt_count"],
                duration=flow_rec["duration"],
                xgb_prob=flow_rec["xgb_prob"],
                ae_score=flow_rec["ae_score"],
                fusion_score=flow_rec["fusion_score"],
                attack_label=flow_rec.get("attack_label", 0),
            )
        return builder

    def evaluate_realtime_gnn_risk(self):
        """
        Rebuilds the sliding-window bipartite graph from the flow buffer, then
        runs Dynamic Bipartite Temporal GNN inference.

        Returns:
            probs      : np.ndarray of shape (num_edges,) — sigmoid risk per edge (classification)
            temporal_shift : float — L2 norm of GRU hidden states (supplementary context)
            pyg_data   : HeteroData — current graph snapshot
            h_map      : dict mapping host_id -> node_index
            s_map      : dict mapping service_id -> node_index
            gnn_latency_ms: float — GNN inference wall-clock time in milliseconds

        Output convention:
        - probs is the ONLY GNN-produced classification signal (risk probability)
        - temporal_shift is supplementary context from GNN hidden state magnitude
        - Both are exposed to Person 2 for fusion alongside upstream scores
        """
        with self._lock:
            if not self.flow_buffer:
                return np.empty((0,), dtype=np.float32), None, {}, {}, 0.0

            # Rebuild graph from current window to prevent memory growth
            builder = self._rebuild_graph_from_buffer()
            pyg_data, h_map, s_map = builder.to_pyg_hetero()

            edge_type = ("host", "connects_to", "service")
            if edge_type not in pyg_data.edge_types or \
                    pyg_data[edge_type].edge_index.shape[1] == 0:
                return np.empty((0,), dtype=np.float32), pyg_data, h_map, s_map, 0.0

            hx_numpy    = pyg_data["host"].x.numpy()
            sx_numpy    = pyg_data["service"].x.numpy()
            eidx_numpy  = pyg_data[edge_type].edge_index.numpy()
            eattr_numpy = pyg_data[edge_type].edge_attr.numpy()

            hx_scaled    = torch.tensor(self.host_scaler.transform(hx_numpy), dtype=torch.float32)
            sx_scaled    = torch.tensor(self.service_scaler.transform(sx_numpy), dtype=torch.float32)
            eidx         = torch.tensor(eidx_numpy, dtype=torch.long)
            eattr_scaled = torch.tensor(self.edge_scaler.transform(eattr_numpy), dtype=torch.float32)

            # Reset GRU states if graph topology changed (node count changed)
            num_hosts    = hx_scaled.shape[0]
            num_services = sx_scaled.shape[0]
            if (self.prev_host_h is not None and
                    self.prev_host_h.shape[0] != num_hosts):
                logger.debug(
                    "OnlineGraphStream: host node count changed %d -> %d, resetting host GRU state.",
                    self.prev_host_h.shape[0], num_hosts,
                )
                self.prev_host_h = None
            if (self.prev_service_h is not None and
                    self.prev_service_h.shape[0] != num_services):
                logger.debug(
                    "OnlineGraphStream: service node count changed %d -> %d, resetting service GRU state.",
                    self.prev_service_h.shape[0], num_services,
                )
                self.prev_service_h = None

            self.gnn_model.eval()
            t_gnn_start = time.perf_counter()
            with torch.no_grad():
                logits, next_host_h, next_service_h = self.gnn_model(
                    hx_scaled, sx_scaled, eidx, eattr_scaled,
                    self.prev_host_h, self.prev_service_h,
                )
                probs = torch.sigmoid(logits).numpy()
            t_gnn_end = time.perf_counter()
            gnn_latency_ms = (t_gnn_end - t_gnn_start) * 1000.0

            # Compute temporal shift from GRU hidden state L2 norm.
            # This is supplementary context, NOT a classification signal.
            # The GNN's actual classification output is `probs` (sigmoid risk).
            temporal_shift = float(torch.norm(next_host_h).item())

            # Persist GRU memory state for next call
            self.prev_host_h    = next_host_h.detach()
            self.prev_service_h = next_service_h.detach()

            self._prev_num_hosts    = num_hosts
            self._prev_num_services = num_services

        return probs, temporal_shift, pyg_data, h_map, s_map, gnn_latency_ms

    @property
    def buffer_size(self):
        """Current number of flows in the sliding window."""
        with self._lock:
            return len(self.flow_buffer)

    def reset(self):
        """Clears the flow buffer and GRU states. Use when starting a new capture session."""
        with self._lock:
            self.flow_buffer.clear()
            self.prev_host_h    = None
            self.prev_service_h = None
            logger.info("OnlineGraphStream: state reset.")
