"""
features/enriched_features.py
============================
Enriched Behavioral Feature Engineering Layer

Extends the base 49-feature schema with additional behavioral features that capture
inter-arrival timing, TCP flag behavior, and payload/byte-distribution beyond what
is already computed in flow_extractor.py.

Architecture:
  RAW PACKETS / FLOWS
          |
  flow_extractor.py (_FlowRecord) ──► 49 base features ──► trained ML models
          |                                                      |
          └──────────────────────────────────────────────────► enriched features
                                                               │
                                                               └── analysis / logging / temporal layer

The existing trained models (XGBoost, Autoencoder, GNN) continue to use the
base 49 features only. Enriched features are for analysis, logging, and future
research experiments.

Feature categories:
  - IAT:     inter-arrival time statistics (median added beyond base min/max/avg/std)
  - TCP:     per-flag indicators and flag rates
  - PAYLOAD: packet-length statistics (mean, variance, median, entropy proxy)

Mathematical definitions:
  IAT_median = median(Δt_i) where Δt_i = t_i - t_(i-1) for packets i>1 in the flow
  TCP_SYN    = 1 if TCP_FLAGS & SYN else 0  (first packet flag OR over flow lifetime)
  TCP_flag_rate = count(flag_set) / total_packets
  Payload_mean  = mean(packet_lengths)
  Payload_std  = std(packet_lengths)
  Payload_var  = variance(packet_lengths)
  Payload_ent  = Shannon entropy of packet-length distribution (bucket-level)
"""

import math
from typing import Dict, List, Optional, Any, Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TCP Flag Constants (matching flow_extractor.py)
# ─────────────────────────────────────────────────────────────────────────────

TCP_FIN = 0x01
TCP_SYN = 0x02
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_ACK = 0x10
TCP_URG = 0x20


# ─────────────────────────────────────────────────────────────────────────────
# Safe statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(lst: List[float]) -> float:
    """Return mean of a list, or 0.0 if empty."""
    if not lst:
        return 0.0
    return float(sum(lst) / len(lst))


def _safe_std(lst: List[float]) -> float:
    """Return population std of a list, or 0.0 if len < 2."""
    if len(lst) < 2:
        return 0.0
    mean = sum(lst) / len(lst)
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)
    return float(math.sqrt(variance))


def _safe_var(lst: List[float]) -> float:
    """Return population variance of a list, or 0.0 if len < 2."""
    if len(lst) < 2:
        return 0.0
    mean = sum(lst) / len(lst)
    return float(sum((x - mean) ** 2 for x in lst) / len(lst))


def _safe_median(lst: List[float]) -> float:
    """Return median of a list, or 0.0 if empty."""
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    if n % 2 == 0:
        return float((s[n // 2 - 1] + s[n // 2]) / 2.0)
    return float(s[n // 2])


def _shannon_entropy_bucket(counts: List[int]) -> float:
    """
    Compute Shannon entropy of a categorical distribution from bucket counts.

    H = -Σ p_i * log2(p_i)  where p_i = counts[i] / sum(counts)

    Returns 0.0 if counts is empty or all-zero (no variability).
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return float(h)


# ─────────────────────────────────────────────────────────────────────────────
# Per-flag extraction from bitmask (for dataset / already-aggregated flows)
# ─────────────────────────────────────────────────────────────────────────────

def _flags_from_bitmask(bitmask: float) -> Dict[str, float]:
    """
    Extract per-flag presence (0/1) from a TCP flag bitmask.

    Parameters
    ----------
    bitmask : float
        Integer bitmask (e.g. TCP_FLAGS from the dataset).

    Returns
    -------
    dict with keys: TCP_SYN, TCP_ACK, TCP_FIN, TCP_RST, TCP_PSH, TCP_URG
    """
    flags = int(bitmask) if bitmask is not None else 0
    return {
        "TCP_SYN": float(bool(flags & TCP_SYN)),
        "TCP_ACK": float(bool(flags & TCP_ACK)),
        "TCP_FIN": float(bool(flags & TCP_FIN)),
        "TCP_RST": float(bool(flags & TCP_RST)),
        "TCP_PSH": float(bool(flags & TCP_PSH)),
        "TCP_URG": float(bool(flags & TCP_URG)),
    }


def _flag_counts_from_bitmask(bitmask: float) -> Dict[str, float]:
    """
    Return per-flag counts (= 1 if flag present, 0 otherwise) from a bitmask.

    Unlike _flags_from_bitmask, this is identical in result for a single aggregate
    bitmask (the dataset only has the final accumulated bitmask, not per-packet flags).

    Use this for the dataset path where we have the final accumulated TCP_FLAGS.
    """
    return _flags_from_bitmask(bitmask)


# ─────────────────────────────────────────────────────────────────────────────
# IAT enrichment (live packets — adds median beyond base min/max/avg/std)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_iat_stats(iat_list: List[float]) -> Dict[str, float]:
    """
    Compute IAT statistics from a list of inter-arrival times in seconds.

    The base 49 features already include MIN, MAX, AVG, STDDEV in milliseconds.
    This adds MEDIAN which is not derivable from those four statistics alone.

    Parameters
    ----------
    iat_list : list of float
        List of inter-arrival times in seconds (as accumulated by _FlowRecord).

    Returns
    -------
    dict with: iat_median_ms, iat_count
    """
    if not iat_list:
        return {
            "iat_median_ms": 0.0,
            "iat_count": 0,
        }
    median_s = _safe_median(iat_list)
    return {
        "iat_median_ms": float(median_s * 1000.0),  # convert to ms
        "iat_count": len(iat_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Payload / packet-length statistics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_payload_stats(pkt_lengths: List[int]) -> Dict[str, float]:
    """
    Compute packet-length statistics beyond base MIN/MAX.

    The base 49 features already include MIN_IP_PKT_LEN and MAX_IP_PKT_LEN.
    This adds MEAN, STD, VARIANCE, MEDIAN, and Shannon entropy of the bucket
    distribution.

    Parameters
    ----------
    pkt_lengths : list of int
        List of IP packet lengths in bytes (as accumulated by _FlowRecord).

    Returns
    -------
    dict with: payload_mean, payload_std, payload_var, payload_median,
              payload_entropy
    """
    if not pkt_lengths:
        return {
            "payload_mean": 0.0,
            "payload_std": 0.0,
            "payload_var": 0.0,
            "payload_median": 0.0,
            "payload_entropy": 0.0,
        }

    # Shannon entropy of the NF-UNSW packet-length bucket distribution
    # Buckets: 0-128, 128-256, 256-512, 512-1024, 1024-1514
    buckets = [0, 0, 0, 0, 0]
    for length in pkt_lengths:
        if   length <= 128:   buckets[0] += 1
        elif length <= 256:    buckets[1] += 1
        elif length <= 512:    buckets[2] += 1
        elif length <= 1024:   buckets[3] += 1
        else:                  buckets[4] += 1

    return {
        "payload_mean":    _safe_mean(pkt_lengths),
        "payload_std":     _safe_std(pkt_lengths),
        "payload_var":     _safe_var(pkt_lengths),
        "payload_median":  _safe_median(pkt_lengths),
        "payload_entropy": _shannon_entropy_bucket(buckets),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TCP flag rates (fraction of packets with each flag set)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_flag_rates(
    iat_list: List[float],
    tcp_flags_list: List[int],
) -> Dict[str, float]:
    """
    Compute the fraction of packets in a flow that have each TCP flag set.

    Requires per-packet TCP flags (available from live _FlowRecord: we would need
    to store per-packet flags in _FlowRecord to compute this). Since the current
    _FlowRecord only stores accumulated bitmasks, this is marked as unavailable
    for the current implementation and returns zeros.

    To enable: store per-packet flags in _FlowRecord (tcp_flags_list) and pass
    them here. For now, flag rates are computed from the accumulated bitmask
    which gives only presence/absence, not rate.

    Parameters
    ----------
    iat_list : list of float
        List of IAT values (used for packet count)
    tcp_flags_list : list of int
        Per-packet TCP flag bitmask values

    Returns
    -------
    dict with: tcp_syn_rate, tcp_ack_rate, tcp_fin_rate, tcp_rst_rate,
              tcp_psh_rate, tcp_urg_rate
    """
    # Per-packet flag rates require storing flags per-packet in _FlowRecord.
    # The current _FlowRecord only stores accumulated bitmasks.
    # Return zeros with a note that this requires per-packet flag tracking.
    return {
        "tcp_syn_rate": 0.0,
        "tcp_ack_rate": 0.0,
        "tcp_fin_rate": 0.0,
        "tcp_rst_rate": 0.0,
        "tcp_psh_rate": 0.0,
        "tcp_urg_rate": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_flow_record(flow) -> Dict[str, Any]:
    """
    Extract enriched behavioral features from a _FlowRecord instance.

    This is the LIVE path — all per-packet information is available:
    - Per-packet IAT values (iat_in, iat_out)
    - Per-packet TCP flags (accumulated as bitmasks only in current implementation)
    - Per-packet lengths (pkt_lengths)

    Parameters
    ----------
    flow : _FlowRecord
        The internal flow accumulator from flow_extractor.py.

    Returns
    -------
    dict of enriched features (floats). Keys:
      - iat_median_ms : median IAT in ms (base has min/max/avg/std)
      - iat_count_in  : number of IAT samples in forward direction
      - iat_count_out  : number of IAT samples in reverse direction
      - iat_count_total : total IAT samples
      - payload_mean   : mean packet length in bytes
      - payload_std    : std of packet lengths
      - payload_var    : variance of packet lengths
      - payload_median : median packet length
      - payload_entropy : Shannon entropy of packet-length bucket distribution
      - tcp_syn, tcp_ack, tcp_fin, tcp_rst, tcp_psh, tcp_urg
          : 1 if flag was seen in the flow, 0 otherwise (from accumulated bitmask)
      - tcp_flag_mask : the accumulated TCP_FLAGS bitmask value
      - tcp_syn_rate, tcp_ack_rate, tcp_fin_rate, tcp_rst_rate,
        tcp_psh_rate, tcp_urg_rate
          : per-packet flag rates (requires per-packet flag tracking; currently 0)
      - tcp_client_flags, tcp_server_flags : per-direction accumulated bitmask
    """
    # ── IAT enrichment ───────────────────────────────────────────────────
    iat_in_stats  = _compute_iat_stats(flow.iat_in)
    iat_out_stats = _compute_iat_stats(flow.iat_out)

    # ── Payload statistics ─────────────────────────────────────────────────
    payload_stats = _compute_payload_stats(flow.pkt_lengths)

    # ── TCP flag extraction from accumulated bitmasks ─────────────────────
    # The flow extractor stores accumulated bitmasks (OR over all packets).
    # Per-packet flag rates are currently unavailable.
    flags = _flags_from_bitmask(flow.tcp_flags_all)
    flag_rates = _compute_flag_rates(flow.iat_in + flow.iat_out, [])

    return {
        # IAT enrichment
        "iat_median_ms":   (iat_in_stats["iat_median_ms"] + iat_out_stats["iat_median_ms"]) / 2.0,
        "iat_count_in":    iat_in_stats["iat_count"],
        "iat_count_out":   iat_out_stats["iat_count"],
        "iat_count_total": iat_in_stats["iat_count"] + iat_out_stats["iat_count"],

        # Payload statistics
        **payload_stats,

        # TCP flags (presence from accumulated bitmask)
        "tcp_syn": flags["TCP_SYN"],
        "tcp_ack": flags["TCP_ACK"],
        "tcp_fin": flags["TCP_FIN"],
        "tcp_rst": flags["TCP_RST"],
        "tcp_psh": flags["TCP_PSH"],
        "tcp_urg": flags["TCP_URG"],

        # TCP flag bitmasks
        "tcp_flag_mask":     float(flow.tcp_flags_all),
        "tcp_client_flags": float(flow.client_tcp_flags),
        "tcp_server_flags": float(flow.server_tcp_flags),

        # TCP flag rates (requires per-packet flag tracking — currently 0)
        **flag_rates,
    }


def extract_from_dataset_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract enriched behavioral features from a dataset flow record dict.

    This is the DATASET path — only aggregate statistics are available.
    Individual per-packet IAT values are NOT available; IAT median cannot
    be computed. TCP flag presence is extracted from the aggregated bitmask.
    Payload statistics are derived from the existing packet-length bucket features.

    Parameters
    ----------
    record : dict
        A flow record dict with keys matching the base 49-feature schema
        (and optionally _pkt_lengths for extended payload stats).

    Returns
    -------
    dict of enriched features. Keys not derivable from the dataset are
    returned as None and documented in the result.
    """
    tcp_flags = record.get("TCP_FLAGS", 0)
    flags = _flags_from_bitmask(tcp_flags)

    # ── Payload statistics ────────────────────────────────────────────────
    # From packet-length buckets (base 49 features)
    # We can reconstruct approximate packet-length distribution from buckets
    bucket_counts = [
        record.get("NUM_PKTS_UP_TO_128_BYTES", 0),
        record.get("NUM_PKTS_128_TO_256_BYTES", 0),
        record.get("NUM_PKTS_256_TO_512_BYTES", 0),
        record.get("NUM_PKTS_512_TO_1024_BYTES", 0),
        record.get("NUM_PKTS_1024_TO_1514_BYTES", 0),
    ]
    total_pkts = sum(bucket_counts)

    if total_pkts == 0:
        payload_stats = {
            "payload_mean": 0.0,
            "payload_std": 0.0,
            "payload_var": 0.0,
            "payload_median": 0.0,
            "payload_entropy": 0.0,
        }
    else:
        # Reconstruct approximate packet-length list from bucket midpoints
        # Buckets: [0,128], [128,256], [256,512], [512,1024], [1024,1514]
        # Use bucket midpoint as approximate length
        bucket_midpoints = [64, 192, 384, 768, 1269]
        reconstructed_lengths = []
        for count, midpoint in zip(bucket_counts, bucket_midpoints):
            reconstructed_lengths.extend([midpoint] * int(count))

        payload_stats = _compute_payload_stats(reconstructed_lengths)

    # ── IAT — median NOT derivable from dataset aggregates ───────────────
    # The dataset has SRC_TO_DST_IAT_MIN/MAX/AVG/STDDEV but not individual IAT values.
    # IAT_median is None (not available from dataset).
    # IAT statistics from base features (already available, included for completeness):
    iat_base = {
        "iat_src_dst_min":   record.get("SRC_TO_DST_IAT_MIN", 0.0),
        "iat_src_dst_max":   record.get("SRC_TO_DST_IAT_MAX", 0.0),
        "iat_src_dst_avg":   record.get("SRC_TO_DST_IAT_AVG", 0.0),
        "iat_src_dst_std":   record.get("SRC_TO_DST_IAT_STDDEV", 0.0),
        "iat_dst_src_min":   record.get("DST_TO_SRC_IAT_MIN", 0.0),
        "iat_dst_src_max":   record.get("DST_TO_SRC_IAT_MAX", 0.0),
        "iat_dst_src_avg":   record.get("DST_TO_SRC_IAT_AVG", 0.0),
        "iat_dst_src_std":   record.get("DST_TO_SRC_IAT_STDDEV", 0.0),
    }

    return {
        # IAT enrichment (median NOT available from dataset aggregates)
        "iat_median_ms": None,  # NOT derivable from dataset — individual IAT values unavailable
        "iat_count_in":  None,  # NOT available from dataset
        "iat_count_out": None,  # NOT available from dataset
        "iat_count_total": None,  # NOT available from dataset

        # IAT base features (already in dataset, included for completeness)
        **iat_base,

        # Payload statistics (from bucket reconstruction)
        **payload_stats,

        # TCP flags (from aggregated bitmask)
        "tcp_syn": flags["TCP_SYN"],
        "tcp_ack": flags["TCP_ACK"],
        "tcp_fin": flags["TCP_FIN"],
        "tcp_rst": flags["TCP_RST"],
        "tcp_psh": flags["TCP_PSH"],
        "tcp_urg": flags["TCP_URG"],

        # TCP flag bitmasks
        "tcp_flag_mask":     float(tcp_flags),
        "tcp_client_flags":  float(record.get("CLIENT_TCP_FLAGS", 0)),
        "tcp_server_flags":  float(record.get("SERVER_TCP_FLAGS", 0)),

        # TCP flag rates (NOT available from dataset — requires per-packet flags)
        "tcp_syn_rate": None,
        "tcp_ack_rate": None,
        "tcp_fin_rate": None,
        "tcp_rst_rate": None,
        "tcp_psh_rate": None,
        "tcp_urg_rate": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature schema registry
# ─────────────────────────────────────────────────────────────────────────────

# Canonical list of enriched feature keys produced by extract_from_flow_record().
# These are ADDITIONAL to the base 49 features — they do NOT replace them.
# Keys with value None in the live path indicate features unavailable from dataset.

ENRICHED_FEATURE_KEYS = [
    # IAT enrichment
    "iat_median_ms",
    "iat_count_in",
    "iat_count_out",
    "iat_count_total",
    # IAT base (available from dataset)
    "iat_src_dst_min",
    "iat_src_dst_max",
    "iat_src_dst_avg",
    "iat_src_dst_std",
    "iat_dst_src_min",
    "iat_dst_src_max",
    "iat_dst_src_avg",
    "iat_dst_src_std",
    # Payload statistics
    "payload_mean",
    "payload_std",
    "payload_var",
    "payload_median",
    "payload_entropy",
    # TCP flags (presence)
    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "tcp_psh",
    "tcp_urg",
    # TCP flag bitmasks
    "tcp_flag_mask",
    "tcp_client_flags",
    "tcp_server_flags",
    # TCP flag rates
    "tcp_syn_rate",
    "tcp_ack_rate",
    "tcp_fin_rate",
    "tcp_rst_rate",
    "tcp_psh_rate",
    "tcp_urg_rate",
]

# Total count of enriched features
assert len(ENRICHED_FEATURE_KEYS) == 32, (
    f"ENRICHED_FEATURE_KEYS should have 32 entries, got {len(ENRICHED_FEATURE_KEYS)}"
)
