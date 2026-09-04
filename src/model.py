"""
PyTorch LSTM World Model for Network Attack Forecasting.
Learns temporal network dynamics: P(S(t+1) | S(t-9), ..., S(t))
and multi-task forecasts of attack probability and MITRE stage.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional
from .config import STATE_FEATURES, STAGE_NAMES

class NetworkWorldModel(nn.Module):
    def __init__(
        self,
        input_dim: int = len(STATE_FEATURES),
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_stages: int = len(STAGE_NAMES),
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_stages = num_stages
        
        # Temporal State Dynamics Encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Shared temporal representation projection
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Head 1: World Model Next-State Predictor S(t+1)
        self.next_state_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
        # Head 2: Attack Probability Head P(Attack(t+1))
        self.attack_prob_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Head 3: Attack Stage Classification Head (Logits over MITRE stages)
        self.stage_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_stages)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, seq_len, input_dim)
        Returns:
            next_state: (batch_size, input_dim)
            attack_prob: (batch_size, 1)
            stage_logits: (batch_size, num_stages)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        last_step = lstm_out[:, -1, :]  # Take the representation of the latest temporal state S(t)
        shared = self.shared_fc(last_step)
        
        next_state = self.next_state_head(shared)
        attack_prob = self.attack_prob_head(shared)
        stage_logits = self.stage_head(shared)
        
        return next_state, attack_prob, stage_logits

    def forward_rollout(
        self, 
        initial_seq: torch.Tensor, 
        steps: int = 5,
        device: torch.device = torch.device("cpu")
    ) -> List[Dict]:
        """
        Forward simulation rollout of the network state trajectory:
        S(t-9)...S(t) -> S(t+1) -> S(t+2) -> ... -> S(t+K)
        """
        self.eval()
        trajectory = []
        curr_seq = initial_seq.clone().to(device)
        if curr_seq.ndim == 2:
            curr_seq = curr_seq.unsqueeze(0)  # (1, seq_len, input_dim)
            
        with torch.no_grad():
            for step_idx in range(1, steps + 1):
                next_state, attack_prob, stage_logits = self.forward(curr_seq)
                
                probs = torch.softmax(stage_logits, dim=-1).squeeze(0).cpu().numpy()
                p_attack = float(attack_prob.squeeze().cpu().item())
                
                if p_attack >= 0.35:
                    attack_stage_idx = 1 + int(np.argmax(probs[1:])) if len(probs) > 1 else 0
                    pred_stage = STAGE_NAMES[attack_stage_idx]
                    confidence = float(probs[attack_stage_idx])
                else:
                    pred_stage = "Normal"
                    confidence = float(probs[0])
                
                trajectory.append({
                    "step": step_idx,
                    "horizon_label": f"+{step_idx}",
                    "attack_probability": p_attack,
                    "predicted_stage": pred_stage,
                    "stage_confidence": confidence,
                    "stage_distribution": {STAGE_NAMES[i]: float(probs[i]) for i in range(len(STAGE_NAMES))},
                    "simulated_state": next_state.squeeze(0).cpu().numpy()
                })
                
                # Auto-regressive trajectory rollout: slide sequence and append simulated next_state
                # S(t-8), ..., S(t), S(t+1)
                curr_seq = torch.cat([curr_seq[:, 1:, :], next_state.unsqueeze(1)], dim=1)
                
        return trajectory
