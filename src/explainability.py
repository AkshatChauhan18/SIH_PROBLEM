"""
Sentinel-X Explainability Module
Provides transparent, model-based feature attributions (sensitivity / perturbation analysis)
explaining why Sentinel-X predicts an elevated attack probability.
"""
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from .config import STATE_FEATURES, FEATURE_DISPLAY_NAMES
from .model import NetworkWorldModel

def compute_feature_attributions(
    model: Optional[NetworkWorldModel],
    sequence_features: np.ndarray,
    scaler = None,
    device: Optional[torch.device] = None,
    baseline_stats: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Computes transparent perturbation-based feature attribution:
    Measures the sensitivity of the predicted attack probability to each feature in S(t).
    Attribution = P(Attack | S) - P(Attack | S with feature_i reverted to baseline)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_features = sequence_features.shape[1]
    curr_state = sequence_features[-1].copy()

    # Pre-calculated normal baseline means for CIC-IDS2017 features (used if baseline_stats is None)
    default_baselines = {
        "total_flows": 15.0,
        "total_packets": 40.0,
        "total_bytes": 12000.0,
        "unique_source_ips": 3.0,
        "unique_dest_ips": 4.0,
        "unique_dest_ports": 3.0,
        "port_diversity": 0.15,
        "dest_diversity": 0.20,
        "syn_ratio": 0.02,
        "ack_ratio": 0.40,
        "rst_ratio": 0.005,
        "fin_ratio": 0.01,
        "psh_ratio": 0.15,
        "flow_packets_per_sec": 8.0,
        "flow_bytes_per_sec": 2400.0,
        "flow_iat_mean": 250000.0,
        "flow_iat_std": 100000.0,
        "flow_iat_max": 1000000.0,
        "flow_iat_min": 10.0,
        "packet_length_mean": 120.0,
        "packet_length_std": 150.0,
        "packet_length_variance": 22500.0,
        "average_packet_size": 130.0,
        "down_up_ratio": 0.8,
        "active_mean": 50000.0,
        "idle_mean": 200000.0,
        "init_window_forward_mean": 8192.0,
        "init_window_backward_mean": 8192.0,
    }
    
    baseline_vector = np.array([
        (baseline_stats.get(feat, default_baselines.get(feat, 0.0)) if baseline_stats else default_baselines.get(feat, 0.0))
        for feat in STATE_FEATURES
    ], dtype=np.float32)

    attributions = []

    if model is not None:
        model.eval()
        
        # Prepare unperturbed normalized sequence
        if scaler is not None:
            norm_seq = scaler.transform(sequence_features)
        else:
            norm_seq = sequence_features.copy()
            
        base_tensor = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, base_p, _ = model(base_tensor)
            p_actual = float(base_p.item())

        # Perturb each feature in the final state S(t) to baseline value
        for i, feat_name in enumerate(STATE_FEATURES):
            perturbed_seq = sequence_features.copy()
            perturbed_seq[-1, i] = baseline_vector[i]
            
            if scaler is not None:
                perturbed_norm = scaler.transform(perturbed_seq)
            else:
                perturbed_norm = perturbed_seq.copy()
                
            p_tensor = torch.tensor(perturbed_norm, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, perturbed_p, _ = model(p_tensor)
                p_perturbed = float(perturbed_p.item())

            # Impact: how much the feature increased attack probability over baseline
            impact = p_actual - p_perturbed
            
            attributions.append({
                "feature": feat_name,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
                "actual_value": float(curr_state[i]),
                "baseline_value": float(baseline_vector[i]),
                "contribution": float(impact),
                "abs_contribution": abs(float(impact))
            })
    else:
        # Transparent domain perturbation sensitivity fallback
        # Evaluates distance from benign network baselines
        weights = {
            "syn_ratio": 0.40,
            "port_diversity": 0.35,
            "rst_ratio": 0.25,
            "flow_packets_per_sec": 0.20,
            "flow_iat_std": 0.18,
            "unique_dest_ports": 0.16,
            "packet_length_variance": 0.14,
            "dest_diversity": 0.12,
            "psh_ratio": 0.10,
            "total_flows": 0.08
        }
        for i, feat_name in enumerate(STATE_FEATURES):
            act_val = float(curr_state[i])
            base_val = float(baseline_vector[i])
            w = weights.get(feat_name, 0.05)
            
            # Normalized deviation
            scale = max(abs(base_val), 1.0)
            norm_diff = (act_val - base_val) / scale
            contrib = float(np.tanh(norm_diff * 0.5) * w)
            
            attributions.append({
                "feature": feat_name,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
                "actual_value": act_val,
                "baseline_value": base_val,
                "contribution": contrib,
                "abs_contribution": abs(contrib)
            })

    # Sort descending by absolute contribution
    attributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
    return attributions
