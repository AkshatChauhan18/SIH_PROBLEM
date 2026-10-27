"""
src/forecasting/conformal.py

Conformal Prediction & Calibrated Uncertainty Bounds for Multi-Step Attack Forecasting.
Provides distribution-free finite-sample coverage guarantees:
P(y[t+k] in C(S[t])) >= 1 - alpha

Mitigates SOC alert fatigue by replacing brittle point predictions with
statistically calibrated prediction sets.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from src.preprocessing.normalization import CLASS_INDEX_TO_NAME

class ConformalPredictionEngine:
    """
    Inductive Conformal Prediction (ICP) engine calibrated on validation split non-conformity scores.
    """

    def __init__(self, num_classes: int = 7):
        self.num_classes = num_classes
        self.cal_scores: np.ndarray = np.array([])
        self.is_calibrated: bool = False

    def calibrate(self, val_probs: np.ndarray, val_targets: np.ndarray):
        """
        Calibrates on validation probabilities and true event targets.
        Non-conformity score: s_i = 1 - P(y_true | S_i)
        """
        assert len(val_probs) == len(val_targets), "Mismatched validation probabilities and targets"
        n = len(val_targets)
        scores = []
        for i in range(n):
            true_idx = int(val_targets[i])
            prob_true = float(val_probs[i, true_idx]) if true_idx < val_probs.shape[1] else 0.0
            scores.append(1.0 - prob_true)

        self.cal_scores = np.sort(np.array(scores))
        self.is_calibrated = True

    def get_prediction_set(
        self,
        predicted_probs: np.ndarray,
        alpha: float = 0.10
    ) -> Dict[str, Any]:
        """
        Given predicted class probabilities P(c | S_t), outputs the conformal prediction set
        guaranteeing coverage 1 - alpha.

        Args:
            predicted_probs: 1D array of length num_classes
            alpha: Significance level (e.g. 0.10 for 90% confidence, 0.05 for 95% confidence)
        """
        if not self.is_calibrated or len(self.cal_scores) == 0:
            # Fallback heuristic if uncalibrated
            q_val = 1.0 - alpha
        else:
            n = len(self.cal_scores)
            q_idx = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
            q_idx = min(max(0, q_idx), n - 1)
            q_val = float(self.cal_scores[q_idx])

        # A class c is in the prediction set if (1 - prob(c)) <= q_val  <=>  prob(c) >= 1 - q_val
        threshold = max(0.0, 1.0 - q_val)
        included_indices = [c for c in range(len(predicted_probs)) if predicted_probs[c] >= threshold]

        # If empty due to edge rounding, ensure top-1 is included
        if not included_indices:
            included_indices = [int(np.argmax(predicted_probs))]

        included_names = [CLASS_INDEX_TO_NAME.get(i, f"Class {i}") for i in included_indices]

        return {
            "confidence_level": f"{round((1.0 - alpha) * 100, 1)}%",
            "significance_alpha": alpha,
            "inclusion_probability_threshold": round(threshold, 4),
            "prediction_set_indices": included_indices,
            "prediction_set_classes": included_names,
            "set_size": len(included_indices),
            "is_singleton": len(included_indices) == 1,
            "included_probabilities": {
                CLASS_INDEX_TO_NAME.get(i, f"Class {i}"): round(float(predicted_probs[i]), 4)
                for i in included_indices
            }
        }
