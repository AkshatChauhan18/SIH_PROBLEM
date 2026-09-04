"""
src/forecasting/rollout.py
Genuinely recursive autoregressive K-step rollout engine.
Rollout: z[t] -> predict z[t+1] -> append z[t+1] -> predict z[t+2] -> ... -> predict z[t+K].
Does NOT independently predict future steps directly from t.
"""

from typing import List, Dict, Any
import torch
import numpy as np

from src.models.world_model import WorldModel
from src.models.prediction_head import ThreatHead
from src.preprocessing.normalization import CLASS_INDEX_TO_NAME

class AutoregressiveRolloutEngine:
    """
    Executes recursive K-step future state forecasting using the World Model.
    At each step k in 1..K:
    - Feeds rolling window of past + predicted latent states
    - Obtains predicted latent state z[t+k]
    - Computes event probability distribution & threat score
    """

    def __init__(
        self,
        world_model: WorldModel,
        threat_head: ThreatHead,
        horizon_k: int = 5,
        sequence_length: int = 10,
    ):
        self.world_model = world_model
        self.threat_head = threat_head
        self.horizon_k = horizon_k
        self.sequence_length = sequence_length

    @torch.no_grad()
    def rollout(
        self,
        z_history: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Input:
            z_history: Tensor [N, latent_dim] representing historical states z[t-N+1:t]
        Returns:
            trajectory: List of K forecast step records (k=1..K)
        """
        self.world_model.eval()
        self.threat_head.eval()

        device = z_history.device
        # Ensure shape [1, N, latent_dim]
        curr_seq = z_history.unsqueeze(0).clone()  # [1, N, 128]

        trajectory = []

        for step in range(1, self.horizon_k + 1):
            # 1. World Model predicts next latent state z[t+step]
            # Input strictly depends on prior predictions when step > 1 (RECURSIVE ROLLOUT)
            pred_z_next, _ = self.world_model(curr_seq)  # [1, 128]

            # 2. ThreatHead predicts event probabilities and threat score
            probs, threat_score = self.threat_head.predict_distribution(pred_z_next)
            probs_np = probs.squeeze(0).cpu().numpy()
            threat_val = float(threat_score.squeeze(0).cpu().item())

            # Top predicted class
            pred_class_idx = int(np.argmax(probs_np))
            pred_class_name = CLASS_INDEX_TO_NAME.get(pred_class_idx, "UNKNOWN")
            pred_confidence = float(probs_np[pred_class_idx])

            # Class probability dictionary
            prob_dist = {
                CLASS_INDEX_TO_NAME[i]: float(probs_np[i])
                for i in range(len(probs_np))
            }

            risk_tier = ThreatHead.get_risk_level(threat_val)

            step_record = {
                "step": step,
                "horizon_label": f"t+{step}",
                "predicted_latent_state": pred_z_next.squeeze(0).cpu().numpy(),
                "predicted_class_index": pred_class_idx,
                "predicted_class_name": pred_class_name,
                "confidence": pred_confidence,
                "threat_score": threat_val,
                "risk_tier": risk_tier,
                "class_probabilities": prob_dist,
            }
            trajectory.append(step_record)

            # 3. Recursive update: append predicted z[t+step] to sequence, slide window
            pred_z_reshaped = pred_z_next.unsqueeze(1)  # [1, 1, 128]
            curr_seq = torch.cat([curr_seq[:, 1:, :], pred_z_reshaped], dim=1)

        return trajectory
