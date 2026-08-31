"""
features/
=========
Enriched Behavioral Feature Engineering Layer.

This package adds behavioral features on top of the base 49-feature schema:
  - IAT: inter-arrival time statistics (median beyond base min/max/avg/std)
  - TCP: per-flag indicators and flag rates
  - Payload: packet-length statistics (mean, std, variance, median, entropy)
"""

from features.enriched_features import (
    extract_from_flow_record,
    extract_from_dataset_record,
    ENRICHED_FEATURE_KEYS,
    TCP_FIN,
    TCP_SYN,
    TCP_RST,
    TCP_PSH,
    TCP_ACK,
    TCP_URG,
)

__all__ = [
    "extract_from_flow_record",
    "extract_from_dataset_record",
    "ENRICHED_FEATURE_KEYS",
    "TCP_FIN",
    "TCP_SYN",
    "TCP_RST",
    "TCP_PSH",
    "TCP_ACK",
    "TCP_URG",
]
