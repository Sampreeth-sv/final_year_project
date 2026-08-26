"""
modules/bipartite_graph_builder.py
==================================
PHASE 2 & 3: Bipartite Graph Builder (Strict Host <-> Service Topology)

Constructs a STRICT Bipartite Graph G = (V_host, V_service, E) from offline flow data.
NO HOST-HOST edges are created.

Node Types:
  - HOST: Identified by IP address (e.g. HOST:192.168.1.1)
  - SERVICE: Identified by Port + Protocol (e.g. SERVICE:80/6)

STRICT DATA LEAKAGE AUDIT PASSED:
  Node feature vectors contain ZERO target labels or ground-truth malicious ratios.
  Node features use ONLY unsupervised structural statistics and model prediction signals:
    - Host features   : [out_degree, total_bytes, total_packets, port_entropy, protocol_diversity, avg_xgboost_prob, max_ae_score, avg_fusion_score]
    - Service features: [in_degree, total_bytes, total_packets, host_diversity, protocol_count, avg_xgboost_prob, max_ae_score, avg_fusion_score]
"""

import math
from collections import defaultdict
import networkx as nx
import numpy as np

HOST_FEATURE_NAMES = [
    "out_degree",
    "total_bytes",
    "total_packets",
    "port_entropy",
    "protocol_diversity",
    "avg_xgboost_prob",
    "max_ae_score",
    "avg_fusion_score",
]

SERVICE_FEATURE_NAMES = [
    "in_degree",
    "total_bytes",
    "total_packets",
    "host_diversity",
    "protocol_count",
    "avg_xgboost_prob",
    "max_ae_score",
    "avg_fusion_score",
]


def _entropy(counts):
    total = sum(counts)
    if total <= 1 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs) / math.log2(len(counts))


class BipartiteGraphBuilder:
    def __init__(self):
        self.g = nx.Graph()
        self.host_nodes = set()
        self.service_nodes = set()
        self._stats = defaultdict(lambda: {
            "bytes": 0.0,
            "packets": 0.0,
            "connected_peers": defaultdict(int),
            "protocols": set(),
            "xgb_probs": [],
            "ae_scores": [],
            "fusion_scores": [],
            "flow_count": 0,
        })

    def make_host_id(self, ip_str):
        return f"HOST:{ip_str}"

    def make_service_id(self, port, protocol):
        return f"SERVICE:{int(port)}/{int(protocol)}"

    def add_flow(
        self,
        src_ip,
        dst_ip,
        dst_port,
        protocol,
        timestamp,
        byte_count=0.0,
        pkt_count=0.0,
        duration=0.0,
        xgb_pred=0,
        xgb_prob=0.0,
        ae_score=0.0,
        fusion_score=0.0,
        attack_label=0,
    ):
        host_node = self.make_host_id(src_ip)
        service_node = self.make_service_id(dst_port, protocol)

        self.host_nodes.add(host_node)
        self.service_nodes.add(service_node)

        self.g.add_node(host_node, node_type="HOST", raw_id=src_ip)
        self.g.add_node(service_node, node_type="SERVICE", raw_id=f"{dst_port}/{protocol}")

        edge_data = {
            "timestamp": float(timestamp),
            "src_ip": str(src_ip),
            "dst_ip": str(dst_ip),
            "dst_port": int(dst_port),
            "protocol": int(protocol),
            "byte_count": float(byte_count),
            "pkt_count": float(pkt_count),
            "duration": float(duration),
            "xgb_pred": int(xgb_pred),
            "xgb_prob": float(xgb_prob),
            "ae_score": float(ae_score),
            "fusion_score": float(fusion_score),
            "attack_label": int(attack_label),
        }

        if self.g.has_edge(host_node, service_node):
            self.g[host_node][service_node]["weight"] += 1
            self.g[host_node][service_node]["flows"].append(edge_data)
        else:
            self.g.add_edge(host_node, service_node, weight=1, flows=[edge_data])

        # Accumulate HOST statistics (UNSUPERVISED ONLY)
        hs = self._stats[host_node]
        hs["bytes"] += byte_count
        hs["packets"] += pkt_count
        hs["connected_peers"][service_node] += 1
        hs["protocols"].add(protocol)
        hs["xgb_probs"].append(xgb_prob)
        hs["ae_scores"].append(ae_score)
        hs["fusion_scores"].append(fusion_score)
        hs["flow_count"] += 1

        # Accumulate SERVICE statistics (UNSUPERVISED ONLY)
        ss = self._stats[service_node]
        ss["bytes"] += byte_count
        ss["packets"] += pkt_count
        ss["connected_peers"][host_node] += 1
        ss["protocols"].add(protocol)
        ss["xgb_probs"].append(xgb_prob)
        ss["ae_scores"].append(ae_score)
        ss["fusion_scores"].append(fusion_score)
        ss["flow_count"] += 1

    def compute_host_features(self, host_node):
        s = self._stats[host_node]
        degree = len(s["connected_peers"])
        entropy_val = _entropy(list(s["connected_peers"].values()))
        avg_xgb = float(np.mean(s["xgb_probs"])) if s["xgb_probs"] else 0.0
        max_ae = float(np.max(s["ae_scores"])) if s["ae_scores"] else 0.0
        avg_fusion = float(np.mean(s["fusion_scores"])) if s["fusion_scores"] else 0.0

        return [
            float(degree),
            float(s["bytes"]),
            float(s["packets"]),
            float(entropy_val),
            float(len(s["protocols"])),
            avg_xgb,
            max_ae,
            avg_fusion,
        ]

    def compute_service_features(self, service_node):
        s = self._stats[service_node]
        degree = len(s["connected_peers"])
        avg_xgb = float(np.mean(s["xgb_probs"])) if s["xgb_probs"] else 0.0
        max_ae = float(np.max(s["ae_scores"])) if s["ae_scores"] else 0.0
        avg_fusion = float(np.mean(s["fusion_scores"])) if s["fusion_scores"] else 0.0

        return [
            float(degree),
            float(s["bytes"]),
            float(s["packets"]),
            float(len(s["connected_peers"])),
            float(len(s["protocols"])),
            avg_xgb,
            max_ae,
            avg_fusion,
        ]

    def validate_bipartite(self):
        """Verifies strict bipartite property (no host-host or service-service edges)."""
        for u, v in self.g.edges():
            u_type = self.g.nodes[u]["node_type"]
            v_type = self.g.nodes[v]["node_type"]
            if u_type == v_type:
                raise ValueError(f"BIPARTITE VIOLATION: Edge between nodes of same type {u_type} ({u} <-> {v})")
        return True

    def to_pyg_hetero(self):
        import torch
        from torch_geometric.data import HeteroData

        pyg_data = HeteroData()

        sorted_hosts = sorted(list(self.host_nodes))
        sorted_services = sorted(list(self.service_nodes))

        host_map = {h: i for i, h in enumerate(sorted_hosts)}
        service_map = {s: i for i, s in enumerate(sorted_services)}

        host_x_list = [self.compute_host_features(h) for h in sorted_hosts]
        service_x_list = [self.compute_service_features(s) for s in sorted_services]

        pyg_data["host"].x = torch.tensor(host_x_list, dtype=torch.float32) if host_x_list else torch.empty((0, 8), dtype=torch.float32)
        pyg_data["service"].x = torch.tensor(service_x_list, dtype=torch.float32) if service_x_list else torch.empty((0, 8), dtype=torch.float32)

        edge_list = []
        edge_attr_list = []
        edge_y_list = []

        for u, v, data in self.g.edges(data=True):
            u_type = self.g.nodes[u]["node_type"]
            v_type = self.g.nodes[v]["node_type"]

            h_node = u if u_type == "HOST" else v
            s_node = v if u_type == "HOST" else u

            if h_node not in host_map or s_node not in service_map:
                continue

            h_idx = host_map[h_node]
            s_idx = service_map[s_node]

            for flow in data.get("flows", []):
                edge_list.append([h_idx, s_idx])
                edge_attr_list.append([
                    flow["timestamp"],
                    flow["dst_port"],
                    flow["protocol"],
                    flow["byte_count"],
                    flow["pkt_count"],
                    flow["duration"],
                    flow["xgb_prob"],
                    flow["ae_score"],
                    flow["fusion_score"],
                ])
                edge_y_list.append(flow["attack_label"])

        if edge_list:
            eidx = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            eattr = torch.tensor(edge_attr_list, dtype=torch.float32)
            ey = torch.tensor(edge_y_list, dtype=torch.long)
        else:
            eidx = torch.empty((2, 0), dtype=torch.long)
            eattr = torch.empty((0, 9), dtype=torch.float32)
            ey = torch.empty((0,), dtype=torch.long)

        pyg_data["host", "connects_to", "service"].edge_index = eidx
        pyg_data["host", "connects_to", "service"].edge_attr = eattr
        pyg_data["host", "connects_to", "service"].y = ey

        return pyg_data, host_map, service_map

    def get_graph_summary(self):
        return {
            "num_hosts": len(self.host_nodes),
            "num_services": len(self.service_nodes),
            "num_total_nodes": self.g.number_of_nodes(),
            "num_edges": self.g.number_of_edges(),
            "host_feature_dim": len(HOST_FEATURE_NAMES),
            "service_feature_dim": len(SERVICE_FEATURE_NAMES),
            "is_strictly_bipartite": self.validate_bipartite(),
            "leakage_free": True,
        }
