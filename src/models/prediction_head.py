"""
src/models/prediction_head.py
ThreatHead and multi-class event prediction head.
Maps latent state z[t] in R^128 to future attack event distributions and threat scores.
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.preprocessing.normalization import CLASS_INDEX_TO_NAME

class ThreatHead(nn.Module):
    """
    Predicts operational network event distribution from latent state vector z.
    Also produces threat probability P(Threat) = 1 - P(BENIGN).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns raw unnormalized class logits [B, num_classes]."""
        return self.net(z)

    def predict_distribution(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes softmax probabilities and threat scores.
        Returns:
            probs: [B, num_classes]
            threat_scores: [B] = 1 - probs[:, 0]
        """
        logits = self.forward(z)
        probs = F.softmax(logits, dim=-1)
        threat_scores = 1.0 - probs[:, 0]
        return probs, threat_scores

    @staticmethod
    def get_risk_level(threat_score: float) -> str:
        """Determines SOC risk level tier."""
        if threat_score >= 0.70:
            return "CRITICAL / HIGH"
        elif threat_score >= 0.30:
            return "ELEVATED / MEDIUM"
        return "NOMINAL / LOW"
