"""
src/models/world_model.py
World Model for learning temporal network-state latent dynamics: P(z[t+1] | z[t-N+1:t]).
Uses a GRU to forecast next latent state z[t+1] and couples with ThreatHead under Dual Loss.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gnn_encoder import GNNEncoder
from src.models.prediction_head import ThreatHead

class WorldModel(nn.Module):
    """
    Offline AI World Model that models network state transitions in latent space.
    S[t] -> z[t] -> P(z[t+1] | z[t-N+1:t]) -> Forecast z[t+1:t+K].
    """

    def __init__(
        self,
        latent_dim: int = 128,
        gru_hidden_dim: int = 128,
        gru_num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers

        # GRU modeling temporal evolution of network state
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        # Transition head mapping GRU hidden state to predicted z[t+1]
        self.transition_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            z_seq: Tensor [B, N, latent_dim] - sequence of N recent latent states
            hidden: Optional initial GRU hidden state
        Returns:
            pred_z_next: Tensor [B, latent_dim] - predicted next latent state z[t+1]
            last_hidden: Tensor [num_layers, B, gru_hidden_dim]
        """
        gru_out, last_hidden = self.gru(z_seq, hidden)
        # Last step output
        last_step = gru_out[:, -1, :]  # [B, gru_hidden_dim]
        pred_z_next = self.transition_head(last_step)  # [B, latent_dim]
        return pred_z_next, last_hidden

class NetworkWorldModelSystem(nn.Module):
    """
    Unified end-to-end system combining:
    1. GNN Encoder (Graph S[t] -> z[t])
    2. World Model GRU (z[t-N+1:t] -> predicted z[t+1])
    3. Threat Head (z -> event logits)
    """

    def __init__(
        self,
        gnn_encoder: GNNEncoder,
        world_model: WorldModel,
        threat_head: ThreatHead,
        lambda_threat: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gnn = gnn_encoder
        self.world_model = world_model
        self.threat_head = threat_head
        self.lambda_threat = lambda_threat
        self.class_weights = class_weights

    def encode_graph(self, pyg_data) -> torch.Tensor:
        """Encodes a single or batched PyG Data graph into z."""
        return self.gnn(
            x=pyg_data.x,
            edge_index=pyg_data.edge_index,
            edge_attr=pyg_data.edge_attr,
            batch=getattr(pyg_data, "batch", None),
        )

    def compute_dual_loss(
        self,
        pred_z_next: torch.Tensor,
        actual_z_next: torch.Tensor,
        target_event_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes Enhanced Dual Loss:
            Dynamics Loss = MSE(pred_z, actual_z) + 0.5 * (1 - CosineSimilarity)
            Classification Loss = Multi-Class Focal Loss on actual_z and pred_z rollouts
        """
        target_z = actual_z_next.detach().float()
        pred_z = pred_z_next.float()

        mse_loss = F.mse_loss(pred_z, target_z)
        cos_sim = F.cosine_similarity(pred_z, target_z, dim=-1).clamp(-1.0, 1.0).mean()
        dynamics_loss = mse_loss + 0.5 * (1.0 - cos_sim)

        weight = self.class_weights.to(pred_z_next.device) if self.class_weights is not None else None

        def _focal_ce(logits, targets, gamma=1.5):
            ce = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce).clamp(1e-6, 1.0)
            focal_factor = (1.0 - pt) ** gamma
            if weight is not None:
                focal_factor = weight[targets] * focal_factor
            return (focal_factor * ce).mean()

        # Direct representation supervision on actual next state
        actual_logits = self.threat_head(actual_z_next).float()
        focal_encoder = _focal_ce(actual_logits, target_event_next)

        # Predictive forecast supervision on World Model rollout
        pred_logits = self.threat_head(pred_z_next).float()
        focal_forecast = _focal_ce(pred_logits, target_event_next)

        cls_loss = 0.5 * (focal_encoder + focal_forecast)
        total_loss = dynamics_loss + self.lambda_threat * cls_loss
        return total_loss, dynamics_loss, cls_loss
