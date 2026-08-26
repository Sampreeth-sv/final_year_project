"""
graph/inference_modes.py
========================
Inference Mode Enum for the GNN Pipeline.

Clearly distinguishes three operational modes to avoid confusion between
live capture, deterministic replay, and offline batch evaluation.
"""

from enum import Enum


class InferenceMode(Enum):
    """
    Inference mode for OnlineGraphStream and related components.

    OFFLINE_TRAINING:
        Training on the full NF-UNSW-NB15-v3 dataset in batch mode.
        No packet capture. Data comes from a CSV file.
        Temporal snapshots G_0..G_7 are used.
        Used by: train_temporal_gnn.py, build_temporal_pipeline.py

    OFFLINE_REPLAY:
        Streaming inference on pre-recorded flows in a deterministic order.
        Data comes from replayed flow records (e.g., from the dataset CSV or
        a PCAP replay). NOT "live" — ground truth labels may be available.
        This mode is appropriate for integration tests and CI validation.
        Used by: tests/test_graph_integration.py, evaluate_temporal_gnn.py

    LIVE_CAPTURE:
        Genuine live network inference from Scapy/Npcap packet capture.
        Flows are produced by FlowExtractor.add_packet() / flush_expired().
        Ground truth labels are unavailable (label_enc defaults to 0).
        Requires Npcap or root privileges for packet capture.
        Used by: live_capture.py (base project, NOT this branch)
    """
    OFFLINE_TRAINING = "offline_training"
    OFFLINE_REPLAY   = "offline_replay"
    LIVE_CAPTURE     = "live_capture"


# Human-readable descriptions
MODE_DESCRIPTIONS = {
    InferenceMode.OFFLINE_TRAINING: "Offline batch training on historical dataset",
    InferenceMode.OFFLINE_REPLAY:   "Deterministic replay of pre-recorded flows (not live)",
    InferenceMode.LIVE_CAPTURE:     "Genuine live network capture via Scapy/Npcap",
}


def describe_mode(mode: InferenceMode) -> str:
    """Returns a human-readable description of the inference mode."""
    return MODE_DESCRIPTIONS.get(mode, "Unknown mode")
