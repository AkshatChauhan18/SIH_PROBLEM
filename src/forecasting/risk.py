"""
src/forecasting/risk.py
Future risk quantification and horizon threat/infiltration assessment.
"""

from typing import List, Dict, Any
import numpy as np

def calculate_horizon_risk_summary(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary threat and infiltration risk across the K-step forecast horizon.

    Disclaimers:
    - Independent survival approximation: 1 - prod(1 - p_k).
      Explicitly documented as an upper-bound approximation because future network
      states exhibit temporal autocorrelation.
    - Max step probability: max(p_k) provides a conservative lower-bound.
    """
    if not trajectory:
        return {
            "max_threat_score": 0.0,
            "mean_threat_score": 0.0,
            "overall_risk_level": "NOMINAL / LOW",
            "infiltration_risk_horizon": 0.0,
            "infiltration_risk_formula": "N/A",
        }

    threat_scores = [step["threat_score"] for step in trajectory]
    max_threat = float(max(threat_scores))
    mean_threat = float(np.mean(threat_scores))

    # Infiltration specific probabilities across horizon
    infil_probs = [
        step["class_probabilities"].get("INFILTRATION", 0.0)
        for step in trajectory
    ]
    max_infil_step = float(max(infil_probs))

    # Independent survival approximation
    prod_survival = 1.0
    for p in infil_probs:
        prod_survival *= (1.0 - p)
    approx_horizon_infil = float(1.0 - prod_survival)

    # Determine overall horizon tier
    if max_threat >= 0.70:
        overall_tier = "CRITICAL / HIGH RISK"
    elif max_threat >= 0.30:
        overall_tier = "ELEVATED / MEDIUM RISK"
    else:
        overall_tier = "NOMINAL / LOW RISK"

    return {
        "max_threat_score": round(max_threat, 4),
        "mean_threat_score": round(mean_threat, 4),
        "overall_risk_level": overall_tier,
        "max_step_infiltration_prob": round(max_infil_step, 4),
        "infiltration_risk_horizon": round(approx_horizon_infil, 4),
        "infiltration_risk_formula": "1 - prod(1 - p_k) [Temporally-dependent approximation]",
    }
