import torch
from torch_geometric.data import Data

FEATURE_DIM = 7  # must match modules.graph_builder.FEATURE_NAMES


class GraphDatasetBuilder:
    """Converts a networkx graph (with per-node 'features' attributes,
    as set by GraphBuilder.prepare_graph) into a torch_geometric Data
    object. Falls back to a zero vector for any node missing features
    (e.g. a destination-only host we've never seen originate traffic)."""

    def build(self, graph):
        nodes = list(graph.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}

        edge_index = []
        for u, v in graph.edges():
            edge_index.append([node_index[u], node_index[v]])
            edge_index.append([node_index[v], node_index[u]])

        if len(edge_index) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index).t().contiguous()

        feats = []
        for n in nodes:
            f = graph.nodes[n].get("features")
            feats.append(f if f is not None else [0.0] * FEATURE_DIM)
        x = torch.tensor(feats, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        return data, nodes
