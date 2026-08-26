"""
GraphBuilder — tracks the live host-communication graph and computes
REAL per-host structural features (not the placeholder all-ones vector
the original scaffold used). Feature schema matches train_gnn.py exactly,
so a model trained offline can score this graph without retraining:

    [out_degree, in_degree, total_bytes, total_packets,
     port_entropy, protocol_diversity, retransmit_total]

This is the piece that turns individual flow detections into
COORDINATED multi-device attack detection: a botnet of compromised
devices that each look only mildly suspicious per-flow can still stand
out structurally (many hosts hammering one target, unusual fan-out,
port-scanning entropy patterns) once assembled into a graph.
"""
import math
import time
from collections import defaultdict
import networkx as nx

FEATURE_NAMES = [
    "out_degree", "in_degree", "total_bytes", "total_packets",
    "port_entropy", "protocol_diversity", "retransmit_total",
]


def _entropy(counts):
    total = sum(counts)
    if total <= 1 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs) / math.log2(len(counts))


entropy = _entropy  # public alias


class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self._stats = defaultdict(lambda: {
            "bytes": 0, "packets": 0,
            "dst_ports": defaultdict(int),
            "protocols": set(),
            "retransmit": 0,
        })

    def add_connection(self, src_ip, dst_ip, dst_port=0, protocol=0,
                        total_bytes=0, total_packets=0, retransmit=0):
        now = time.time()
        self.graph.add_node(src_ip)
        self.graph.add_node(dst_ip)
        if self.graph.has_edge(src_ip, dst_ip):
            self.graph[src_ip][dst_ip]["weight"] += 1
        else:
            self.graph.add_edge(src_ip, dst_ip, weight=1)
        self.graph[src_ip][dst_ip]["last_seen"] = now
        self.graph.nodes[src_ip]["last_seen"] = now
        self.graph.nodes[dst_ip]["last_seen"] = now

        s = self._stats[src_ip]
        s["bytes"] += total_bytes
        s["packets"] += total_packets
        s["dst_ports"][dst_port] += 1
        s["protocols"].add(protocol)
        s["retransmit"] += retransmit

    def prune_older_than(self, seconds):
        """Drops edges/nodes not touched within the window, so a
        long-running live deployment's graph doesn't grow unbounded.
        Per-host aggregate stats (self._stats) are intentionally NOT
        reset here -- those are meant to persist as the host's running
        profile; only graph structure is windowed."""
        now = time.time()
        stale_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                       if now - d.get("last_seen", now) > seconds]
        self.graph.remove_edges_from(stale_edges)
        stale_nodes = [n for n in self.graph.nodes()
                       if self.graph.degree[n] == 0
                       and now - self.graph.nodes[n].get("last_seen", now) > seconds]
        self.graph.remove_nodes_from(stale_nodes)

    def get_graph(self):
        return self.graph

    def node_features(self, ip):
        s = self._stats[ip]
        degree = self.graph.degree[ip] if ip in self.graph else 0
        return [
            float(degree),
            float(degree),  # undirected graph; kept as 2 slots to match train_gnn.py schema
            float(s["bytes"]),
            float(s["packets"]),
            _entropy(list(s["dst_ports"].values())),
            float(len(s["protocols"])),
            float(s["retransmit"]),
        ]

    def prepare_graph(self):
        """Builds a torch_geometric Data object over the CURRENT graph,
        with real per-node feature vectors attached. Returns
        (data, node_order) where node_order[i] is the IP at row i."""
        from modules.graph_dataset import GraphDatasetBuilder

        for ip in self.graph.nodes():
            self.graph.nodes[ip]["features"] = self.node_features(ip)

        builder = GraphDatasetBuilder()
        data, node_order = builder.build(self.graph)
        return data, node_order
