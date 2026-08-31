"""
calibrate_temporal_norm.py
==========================
Calibrate the temporal score normalization factor from validation data.

Procedure:
  1. Load the trained GNN model and the temporal graph dataset.
  2. Replay the validation snapshots through the model in temporal order.
  3. After each snapshot, compute the L2 norm of the GRU hidden states.
  4. Collect all L2 norms (temporal_shift values).
  5. Calculate the 95th percentile as the calibrated max_norm.
  6. Save max_norm to models/fusion_config.json.

This calibration uses the VALIDATION set only (temporal_graph_dataset.pt).
The test set is NOT used.

Output:
  - models/fusion_config.json (updated with temporal_max_norm)
  - Console report with calibration statistics
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnn.dynamic_temporal_gnn import DynamicBipartiteTemporalGNN
from config import FUSION_CONFIG_PATH

# ── Model architecture (must match the saved checkpoint) ───────────────────────
# Determined from inspecting dynamic_temporal_gnn.pt state dict:
#   conv1.*.weight: [32, 8]  → host_dim=8, service_dim=8
#   classifier.mlp.0.weight: [32, 73]  → 73 = 32*2 + 9  → edge_dim=9
#   temporal_update.*.weight_ih: [96, 32]  → GRUCell(32, 32)  → hidden_dim=32
HOST_DIM = 8
SERVICE_DIM = 8
EDGE_DIM = 9
HIDDEN_DIM = 32


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(path="temporal_graph_dataset.pt"):
    """Load the temporal graph dataset."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Dataset loaded: {data['num_snapshots']} snapshots")
    return data


# ── Model loader ──────────────────────────────────────────────────────────────

def load_gnn_model(path="dynamic_temporal_gnn.pt"):
    """Load the trained GNN model."""
    model = DynamicBipartiteTemporalGNN(
        host_dim=HOST_DIM,
        service_dim=SERVICE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"GNN model loaded: hidden_dim={HIDDEN_DIM}")
    return model


# ── Snapshot-to-tensor conversion ──────────────────────────────────────────────

def snapshot_to_tensors(snap):
    """
    Convert a dataset snapshot dict to model input tensors.

    The dataset stores:
      host_x: shape (num_hosts, 8)
      service_x: shape (num_services, 8)
      edge_index_host_service: shape (2, num_edges)
      edge_attr: shape (num_edges, 9)
    """
    host_x = torch.tensor(snap["host_x"], dtype=torch.float32)
    service_x = torch.tensor(snap["service_x"], dtype=torch.float32)
    edge_index = torch.tensor(snap["edge_index_host_service"], dtype=torch.long)
    edge_attr = torch.tensor(snap["edge_attr"], dtype=torch.float32)
    return host_x, service_x, edge_index, edge_attr


# ── Calibrate temporal normalization ──────────────────────────────────────────

def calibrate_temporal_norm(model, data):
    """
    Replay validation snapshots through the model and collect L2 norms of GRU states.

    The GRU maintains hidden states that are updated at each snapshot.
    We collect the L2 norm of the host GRU hidden state after each update.
    This reflects the temporal activation level at each snapshot.

    Returns:
        norms: list of L2 norms (one per snapshot)
    """
    norms = []
    prev_host_h = None
    prev_service_h = None

    snapshots = data["snapshots"]
    print(f"Processing {len(snapshots)} snapshots...")

    for i, snap in enumerate(snapshots):
        host_x, service_x, edge_index, edge_attr = snapshot_to_tensors(snap)

        if host_x.shape[0] == 0 or service_x.shape[0] == 0 or edge_index.shape[1] == 0:
            # Skip empty snapshots (no edges)
            print(f"  Snapshot {i}: empty graph, skipping")
            continue

        with torch.no_grad():
            logits, next_host_h, next_service_h = model(
                host_x, service_x, edge_index, edge_attr,
                prev_host_h, prev_service_h,
            )

        # Compute L2 norm of host GRU hidden state
        host_h_norm = float(torch.norm(next_host_h).item())

        # Also track service state norm for completeness
        service_h_norm = float(torch.norm(next_service_h).item())

        norms.append(host_h_norm)
        print(f"  Snapshot {i}: host_norm={host_h_norm:.4f}  service_norm={service_h_norm:.4f}")

        # Propagate GRU state to next snapshot
        prev_host_h = next_host_h.detach()
        prev_service_h = next_service_h.detach()

    return norms


# ── Compute statistics ────────────────────────────────────────────────────────

def compute_statistics(norms):
    """Compute calibration statistics from collected norms."""
    norms_arr = np.array(norms)

    stats = {
        "n_samples": int(len(norms)),
        "min": float(np.min(norms_arr)),
        "max": float(np.max(norms_arr)),
        "mean": float(np.mean(norms_arr)),
        "std": float(np.std(norms_arr)),
        "percentile_50": float(np.percentile(norms_arr, 50)),
        "percentile_90": float(np.percentile(norms_arr, 90)),
        "percentile_95": float(np.percentile(norms_arr, 95)),
        "percentile_99": float(np.percentile(norms_arr, 99)),
    }
    return stats


# ── Save to fusion_config.json ────────────────────────────────────────────────

def save_calibration(config_path, max_norm):
    """Update fusion_config.json with the calibrated temporal_max_norm."""
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Preserve existing values
    config["fusion_threshold"] = config.get("fusion_threshold", 0.5)
    config["temporal_max_norm"] = round(max_norm, 6)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nCalibration saved to {config_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Temporal Score Normalization Calibration")
    print("=" * 60)
    print()

    # Load model and dataset
    print("Loading GNN model...")
    model = load_gnn_model("dynamic_temporal_gnn.pt")

    print("\nLoading temporal graph dataset...")
    data = load_dataset("temporal_graph_dataset.pt")

    # Run calibration
    print("\nCalibrating temporal normalization...")
    print("-" * 60)
    norms = calibrate_temporal_norm(model, data)

    if not norms:
        print("\nERROR: No valid temporal norms collected. Cannot calibrate.")
        sys.exit(1)

    # Compute statistics
    stats = compute_statistics(norms)

    # Use 95th percentile as max_norm
    max_norm = stats["percentile_95"]

    # Report
    print()
    print("=" * 60)
    print("CALIBRATION REPORT")
    print("=" * 60)
    print(f"  Validation samples (snapshots): {stats['n_samples']}")
    print(f"  Minimum temporal norm:          {stats['min']:.6f}")
    print(f"  Maximum temporal norm:          {stats['max']:.6f}")
    print(f"  Mean temporal norm:             {stats['mean']:.6f}")
    print(f"  Std temporal norm:              {stats['std']:.6f}")
    print(f"  50th percentile (median):       {stats['percentile_50']:.6f}")
    print(f"  90th percentile:               {stats['percentile_90']:.6f}")
    print(f"  95th percentile (CALIBRATED):   {stats['percentile_95']:.6f}")
    print(f"  99th percentile:               {stats['percentile_99']:.6f}")
    print()
    print(f"  FINAL CALIBRATED max_norm:      {max_norm:.6f}")
    print()
    print(f"  Rationale: max_norm is set to the 95th percentile so that")
    print(f"  95% of real temporal scores map to [0, 1] range.")
    print("=" * 60)

    # Save
    save_calibration(FUSION_CONFIG_PATH, max_norm)

    # Also print a comparison with the old hardcoded value
    old_default = 100.0
    print(f"\n  Comparison:")
    print(f"    Old hardcoded max_norm: {old_default:.4f}")
    print(f"    New calibrated max_norm: {max_norm:.6f}")
    print(f"    Ratio (new/old):       {max_norm / old_default:.4f}")

    return max_norm


if __name__ == "__main__":
    main()
