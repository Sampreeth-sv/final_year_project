"""
modules/dynamic_temporal_gnn.py
===============================
PHASE 5: Dynamic Bipartite Temporal Graph Neural Network

Architecture:
  1. HeteroBipartiteConv  — mean-aggregated bipartite message passing (Host <-> Service)
  2. TemporalNodeUpdate   — GRU-based recurrent state (h_{t-1} -> h_t) per node type
  3. EdgeRiskClassifier   — MLP edge classifier (host_state || service_state || edge_feat)

Inference note:
  BatchNorm1d in EdgeRiskClassifier will fail with batch_size=1 in train() mode.
  The model must be called with model.eval() during single-flow live inference.
  This is enforced by OnlineGraphStream.evaluate_realtime_gnn_risk().

Training note:
  The GRU hidden states prev_host_h / prev_service_h are passed between snapshot
  steps and detached after each backward pass to prevent gradient accumulation
  across the full sequence. See train_temporal_gnn.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroBipartiteConv(nn.Module):
    """
    Bipartite Graph Convolution aggregating Host <-> Service messages.

    For each edge (h -> s):
      - Host messages are aggregated into service nodes (mean over in-neighbors)
      - Service messages are aggregated into host nodes (mean over in-neighbors)
    Self-connections are added via separate linear transforms.
    """

    def __init__(self, host_dim, service_dim, hidden_dim):
        super().__init__()
        self.host_to_service = nn.Linear(host_dim, hidden_dim)
        self.service_to_host = nn.Linear(service_dim, hidden_dim)
        self.host_self       = nn.Linear(host_dim, hidden_dim)
        self.service_self    = nn.Linear(service_dim, hidden_dim)

    def forward(self, host_x, service_x, edge_index):
        num_hosts    = host_x.shape[0]
        num_services = service_x.shape[0]
        hidden_dim   = self.host_self.out_features

        msg_to_service = torch.zeros((num_services, hidden_dim), device=host_x.device)
        msg_to_host    = torch.zeros((num_hosts,    hidden_dim), device=host_x.device)

        if edge_index.shape[1] > 0:
            h_idx, s_idx = edge_index[0], edge_index[1]

            h_trans = F.relu(self.host_to_service(host_x))
            s_trans = F.relu(self.service_to_host(service_x))

            # Aggregate messages with mean normalization
            msg_to_service.index_add_(0, s_idx, h_trans[h_idx])
            msg_to_host.index_add_(0,    h_idx, s_trans[s_idx])

            s_deg = torch.zeros((num_services, 1), device=host_x.device)\
                         .index_add_(0, s_idx, torch.ones((len(s_idx), 1), device=host_x.device))\
                         .clamp(min=1.0)
            h_deg = torch.zeros((num_hosts, 1), device=host_x.device)\
                         .index_add_(0, h_idx, torch.ones((len(h_idx), 1), device=host_x.device))\
                         .clamp(min=1.0)

            msg_to_service = msg_to_service / s_deg
            msg_to_host    = msg_to_host    / h_deg

        out_host    = F.relu(self.host_self(host_x)    + msg_to_host)
        out_service = F.relu(self.service_self(service_x) + msg_to_service)

        return out_host, out_service


class TemporalNodeUpdate(nn.Module):
    """
    GRU Temporal Recurrent Unit updating node memory across graph snapshots.

    Maintains separate GRU cells for host nodes and service nodes.
    Hidden states are passed between snapshot time steps externally.
    When the number of nodes changes between calls (e.g., after window rebuild),
    the caller is responsible for resetting the hidden state to None.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.host_gru    = nn.GRUCell(hidden_dim, hidden_dim)
        self.service_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, curr_host, curr_service, prev_host_h=None, prev_service_h=None):
        # Initialize hidden states to zero if None or shape-mismatched
        if prev_host_h is None or prev_host_h.shape != curr_host.shape:
            prev_host_h = torch.zeros_like(curr_host)
        if prev_service_h is None or prev_service_h.shape != curr_service.shape:
            prev_service_h = torch.zeros_like(curr_service)

        next_host_h    = self.host_gru(curr_host,    prev_host_h)
        next_service_h = self.service_gru(curr_service, prev_service_h)

        return next_host_h, next_service_h


class EdgeRiskClassifier(nn.Module):
    """
    Edge Classification MLP predicting evolving attack probability per flow edge.

    Input: concat(host_state, service_state, edge_attr)
    Output: scalar logit per edge (pass through sigmoid for probability)

    IMPORTANT: BatchNorm1d requires batch_size >= 2 in train() mode.
    Always call model.eval() for single-flow live inference.
    """

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        in_dim = hidden_dim * 2 + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, host_states, service_states, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return torch.empty((0,), device=host_states.device)

        h_idx = edge_index[0]
        s_idx = edge_index[1]
        h_emb = host_states[h_idx]
        s_emb = service_states[s_idx]

        cat_feat = torch.cat([h_emb, s_emb, edge_attr], dim=1)
        logits   = self.mlp(cat_feat).squeeze(-1)
        return logits


class DynamicBipartiteTemporalGNN(nn.Module):
    """
    Master Dynamic Bipartite Temporal GNN.

    Forward pass:
      1. HeteroBipartiteConv: encode current snapshot into host/service embeddings
      2. TemporalNodeUpdate : update GRU hidden states with current embeddings
      3. EdgeRiskClassifier : predict attack probability per edge using updated states

    Args:
        host_dim    : feature dimension of host node vectors
        service_dim : feature dimension of service node vectors
        edge_dim    : feature dimension of edge attribute vectors
        hidden_dim  : hidden dimension for all intermediate layers

    Forward signature:
        (host_x, service_x, edge_index, edge_attr, prev_host_h, prev_service_h)
        -> (logits, next_host_h, next_service_h)
    """

    def __init__(self, host_dim=8, service_dim=8, edge_dim=9, hidden_dim=32):
        super().__init__()
        self.conv1          = HeteroBipartiteConv(host_dim, service_dim, hidden_dim)
        self.temporal_update = TemporalNodeUpdate(hidden_dim)
        self.classifier     = EdgeRiskClassifier(hidden_dim, edge_dim)

    def forward(self, host_x, service_x, edge_index, edge_attr,
                prev_host_h=None, prev_service_h=None):
        # 1. Bipartite message passing
        h_conv, s_conv = self.conv1(host_x, service_x, edge_index)
        # 2. Temporal GRU state update
        h_state, s_state = self.temporal_update(h_conv, s_conv, prev_host_h, prev_service_h)
        # 3. Edge risk classification
        logits = self.classifier(h_state, s_state, edge_index, edge_attr)
        return logits, h_state, s_state
