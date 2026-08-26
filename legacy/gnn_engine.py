"""
gnn_engine.py
=============
Was an empty (0-byte) stub in the original scaffold. Wraps loading and
running the trained GAT (modules/gat.py) over the live host-communication
graph (modules/graph_builder.py), so inference.py doesn't need to know
the model's architecture details or the graph's feature schema.
"""
import os
import json
import logging
import joblib
import torch
import torch.nn.functional as F

from modules.gat import GATModel

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class GNNEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.loaded = False

    def load(self):
        art_path = os.path.join(_MODELS_DIR, "gat_artifacts.json")
        model_path = os.path.join(_MODELS_DIR, "gat_model.pt")
        scaler_path = os.path.join(_MODELS_DIR, "graph_feature_scaler.pkl")

        if not (os.path.exists(art_path) and os.path.exists(model_path)
                and os.path.exists(scaler_path)):
            logger.warning("GAT artifacts not found — run train_gnn.py first. "
                            "Graph-based coordinated-attack scoring will return 0.0 until then.")
            return

        with open(art_path) as f:
            art = json.load(f)

        self.model = GATModel(input_dim=art["input_dim"], hidden_dim=art["hidden_dim"], output_dim=2)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.loaded = True
        logger.info("GAT engine loaded.")

    def score_graph(self, graph_builder):
        """Runs the trained GAT over the CURRENT live host graph.
        Returns {ip: attack_probability}. Empty dict if not loaded or
        the graph has no nodes yet."""
        if not self.loaded or graph_builder.graph.number_of_nodes() == 0:
            return {}

        data, node_order = graph_builder.prepare_graph()
        x_scaled = self.scaler.transform(data.x.numpy())
        data.x = torch.tensor(x_scaled, dtype=torch.float)

        with torch.no_grad():
            out = self.model(data)
            probs = F.softmax(out, dim=1)[:, 1].numpy()

        return {ip: float(p) for ip, p in zip(node_order, probs)}
