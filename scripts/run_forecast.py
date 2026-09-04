#!/usr/bin/env python3
"""
scripts/run_forecast.py

CLI Demonstration Engine for Multi-Step Attack Forecasting.
Flow: Historical States -> Current State S[t] -> World Model -> Future Trajectory (t+1..t+K) -> Actual Future S[t+1..t+K]
Displays Model Forecast vs Actual Future, MITRE ATT&CK interpretations, and gradient attributions.
"""

import sys
import argparse
from pathlib import Path
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.device import get_device
from src.models.gnn_encoder import GNNEncoder
from src.models.prediction_head import ThreatHead
from src.models.world_model import WorldModel, NetworkWorldModelSystem
from src.forecasting.rollout import AutoregressiveRolloutEngine
from src.forecasting.risk import calculate_horizon_risk_summary
from src.forecasting.mitre_mapping import interpret_prediction_as_mitre
from src.explainability.attribution import compute_temporal_and_feature_attribution
from src.preprocessing.normalization import CLASS_INDEX_TO_NAME

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Run multi-step attack forecast rollout demo.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--window-idx", type=int, default=15, help="Starting test window index t")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon K steps ahead")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["training"]["device"])

    graphs_dir = Path(config["paths"]["graphs_dir"])
    world_model_path = Path(config["paths"]["world_model_path"])

    seq_len = config["temporal_windowing"]["sequence_length"]
    horizon_k = args.horizon or config["temporal_windowing"]["forecast_horizon"]
    win_sec = config["temporal_windowing"]["window_seconds"]
    class_names = config["threat_head"]["classes"]

    # 1. Load Test Graphs
    test_graphs_path = graphs_dir / "test_graphs.pt"
    if not test_graphs_path.is_file():
        print(f"[!] Test graphs not found at: {test_graphs_path}. Run build_graphs.py first.", file=sys.stderr)
        sys.exit(1)

    test_graphs = torch.load(test_graphs_path, weights_only=False)

    # 2. Load World Model System
    if not world_model_path.is_file():
        print(f"[!] Trained World Model not found at: {world_model_path}. Run train_world_model.py first.", file=sys.stderr)
        sys.exit(1)

    checkpoint = torch.load(world_model_path, map_location=device, weights_only=False)
    gnn_encoder = GNNEncoder(
        node_dim=config["graph_construction"]["node_feature_dim"],
        edge_dim=config["graph_construction"]["edge_feature_dim"],
        hidden_dim=config["model_architecture"]["gnn_hidden_dim"],
    )
    world_model = WorldModel(
        latent_dim=config["model_architecture"]["latent_dim"],
        gru_hidden_dim=config["model_architecture"]["gru_hidden_dim"],
        gru_num_layers=config["model_architecture"]["gru_num_layers"],
    )
    threat_head = ThreatHead(
        latent_dim=config["model_architecture"]["latent_dim"],
        hidden_dim=config["threat_head"]["hidden_dim"],
        num_classes=len(class_names),
    )
    system = NetworkWorldModelSystem(gnn_encoder, world_model, threat_head).to(device)
    system.load_state_dict(checkpoint["model_state_dict"])
    system.eval()

    rollout_engine = AutoregressiveRolloutEngine(
        world_model=world_model,
        threat_head=threat_head,
        horizon_k=horizon_k,
        sequence_length=seq_len,
    )

    t_idx = args.window_idx
    if t_idx < seq_len or (t_idx + horizon_k) >= len(test_graphs):
        t_idx = seq_len

    current_graph = test_graphs[t_idx]
    print("=" * 85)
    print("AI WORLD MODEL ATTACK FORECASTING — SOC TELEMETRY DEMONSTRATION")
    print("=" * 85)
    print(f"Current Time (S[t])     : Window #{t_idx} ({getattr(current_graph, 'start_time_iso', 'N/A')})")
    print(f"Current Network State   : Hosts |V| = {current_graph.num_nodes} | Active Edges |E| = {current_graph.edge_index.size(1)}")
    print(f"Current State Label     : {CLASS_INDEX_TO_NAME.get(int(current_graph.y.item()), 'UNKNOWN')}")
    print(f"Forecast Horizon        : K = {horizon_k} steps ({horizon_k * win_sec}s advance lead time)\n")

    # Encode historical sequence z[t-N+1:t]
    with torch.no_grad():
        history_graphs = test_graphs[t_idx - seq_len + 1 : t_idx + 1]
        history_z = torch.cat([system.encode_graph(g.to(device)) for g in history_graphs], dim=0)

    # Execute Recursive Rollout
    trajectory = rollout_engine.rollout(history_z)
    risk_summary = calculate_horizon_risk_summary(trajectory)

    print("-" * 85)
    print(f"{'Horizon':<8} {'Lead Time':<12} {'Predicted Event':<20} {'Conf':>8} {'Threat Prob':>12} {'Actual Future':<18}")
    print("-" * 85)

    for step_rec in trajectory:
        k = step_rec["step"]
        actual_future_graph = test_graphs[t_idx + k]
        actual_name = CLASS_INDEX_TO_NAME.get(int(actual_future_graph.y.item()), "UNKNOWN")

        lead_str = f"+{k * win_sec}s"
        pred_name = step_rec["predicted_class_name"]
        conf = step_rec["confidence"]
        threat = step_rec["threat_score"]

        match_marker = "[MATCH]" if pred_name == actual_name else "[MISMATCH]"
        print(f"{step_rec['horizon_label']:<8} {lead_str:<12} {pred_name:<20} {conf:>7.1%} {threat:>11.1%} {actual_name} {match_marker}")
    print("-" * 85)

    print("\n" + "=" * 85)
    print("FUTURE RISK ASSESSMENT & HORIZON INFILTRATION RISK")
    print("=" * 85)
    print(f"Overall Horizon Risk Tier : {risk_summary['overall_risk_level']}")
    print(f"Max Threat Probability    : {risk_summary['max_threat_score']:.1%}")
    print(f"Horizon Infiltration Risk : {risk_summary['infiltration_risk_horizon']:.1%} ({risk_summary['infiltration_risk_formula']})")

    # MITRE ATT&CK Interpretation for immediate next step t+1
    next_step = trajectory[0]
    mitre_info = interpret_prediction_as_mitre(
        next_step["predicted_class_name"], next_step["confidence"], next_step["threat_score"]
    )
    m_att = mitre_info["mitre_attack_interpretation"]

    print("\n" + "=" * 85)
    print("MITRE ATT&CK INTERPRETATION LAYER (Next State t+1)")
    print("=" * 85)
    print(f"MITRE Tactic        : {m_att['tactic']} ({m_att['tactic_id']})")
    print(f"MITRE Technique     : {m_att['technique']} ({m_att['technique_id']})")
    print(f"Threat Intelligence : {m_att['security_description']}")
    print(f"Recommended Action  : {m_att['recommended_soc_action']}")

    # Explainability & Attribution
    attr = compute_temporal_and_feature_attribution(system, history_z, graph_data=current_graph.to(device))
    print("\n" + "=" * 85)
    print("EXPLAINABILITY: TOP CONTRIBUTING INPUT FEATURES (Gradient Attribution)")
    print("=" * 85)
    for idx, item in enumerate(attr["top_contributing_features"][:5], 1):
        bar = "█" * int(item["importance_score"] * 40)
        print(f"  {idx}. {item['feature']:<30} {item['importance_score']:>6.1%}  {bar}")

    print("\n" + "=" * 85)
    print("TEMPORAL ATTRIBUTION (Historical State Influence over Trajectory)")
    print("=" * 85)
    for lbl, score in zip(attr["temporal_timeline"], attr["temporal_importance"]):
        bar = "█" * int(score * 40)
        print(f"  {lbl:<16} {score:>6.1%}  {bar}")
    print("=" * 85)
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)

if __name__ == "__main__":
    main()
