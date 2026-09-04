"""
src/models/gnn_encoder.py
2-layer GINEConv Graph Neural Network Encoder.
Encodes arbitrary-sized network graphs S[t] into a fixed 128-dimensional latent vector z[t].
Global Mean Pool (64d) + Global Max Pool (64d) = 128d.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

class GNNEncoder(nn.Module):
    """
    Encodes network state graph S[t] into a fixed latent representation z[t] in R^128.
    Invariant to the number of nodes / hosts in the network.
    """

    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 7,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Edge feature projection to match hidden_dim
        self.edge_proj1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edge_proj2 = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
        )

        # Layer 1 MLP: node_dim -> hidden_dim
        mlp1 = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv1 = GINEConv(mlp1, edge_dim=hidden_dim)

        # Layer 2 MLP: hidden_dim -> hidden_dim
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv2 = GINEConv(mlp2, edge_dim=hidden_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass:
        Inputs:
            x: Node features [N, node_dim]
            edge_index: Graph topology [2, E]
            edge_attr: Edge features [E, edge_dim]
            batch: Batch vector assigning each node to a graph [N]
        Returns:
            z: Latent vector [B, 128] (where B = number of graphs in batch)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle empty edges case
        if edge_index.size(1) == 0 or edge_attr.size(0) == 0:
            # Fallback for graph with no active edges
            h1 = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
            h2 = h1
        else:
            e1 = self.edge_proj1(edge_attr)
            h1 = self.conv1(x, edge_index, edge_attr=e1)
            h1 = self.act(h1)
            h1 = self.dropout(h1)

            e2 = self.edge_proj2(edge_attr)
            h2 = self.conv2(h1, edge_index, edge_attr=e2)
            h2 = self.act(h2)

        # Dual pooling: Global Mean Pool (64d) concatenated with Global Max Pool (64d)
        mean_pool = global_mean_pool(h2, batch)  # [B, 64]
        max_pool = global_max_pool(h2, batch)    # [B, 64]

        # Fixed 128-dimensional latent vector z[t]
        z = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 128]
        return z
