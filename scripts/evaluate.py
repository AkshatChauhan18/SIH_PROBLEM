#!/usr/bin/env python3
"""
scripts/evaluate.py

Evaluates GNN + GRU World Model vs Baseline Logistic Regression across multi-step horizons (t+1 ... t+K).
Metrics: Macro F1, Per-Class F1, Precision, Recall, False Positive Rate (FPR), and Forecast Lead Time.
Generates comprehensive comparative JSON report to reports/evaluation_report.json.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.metrics import compute_evaluation_metrics, compute_lead_time_gain
from src.models.gnn_encoder import GNNEncoder
from src.models.prediction_head import ThreatHead
from src.models.world_model import WorldModel, NetworkWorldModelSystem
from src.models.baseline import BaselineLogisticRegression
from src.forecasting.rollout import AutoregressiveRolloutEngine

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Evaluate World Model vs Baseline across K-step horizons.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", type=str, default="reports/evaluation_report.json", help="Report output path")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["training"]["device"])

    graphs_dir = Path(config["paths"]["graphs_dir"])
    world_model_path = Path(config["paths"]["world_model_path"])
    baseline_path = Path(config["paths"]["baseline_model_path"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_len = config["temporal_windowing"]["sequence_length"]
    horizon_k = config["temporal_windowing"]["forecast_horizon"]
    win_sec = config["temporal_windowing"]["window_seconds"]
    class_names = config["threat_head"]["classes"]
    num_classes = len(class_names)

    print("=" * 80)
    print("SIH 2026: MULTI-STEP FORECAST EVALUATION BENCHMARK")
    print("=" * 80)
    print(f"Device             : {device}")
    print(f"Forecast Horizon K : {horizon_k} steps ({horizon_k * win_sec}s advance lead time)")
    print(f"Test Graph Dataset : {graphs_dir / 'test_graphs.pt'}\n")

    # 1. Load Models and Test Graphs
    test_graphs_path = graphs_dir / "test_graphs.pt"
    if not test_graphs_path.is_file():
        print(f"[!] Test graphs not found: {test_graphs_path}. Run build_graphs.py first.", file=sys.stderr)
        sys.exit(1)

    test_graphs = torch.load(test_graphs_path, weights_only=False)
    print(f"[1/3] Loaded {len(test_graphs)} TEST graph windows.")

    # Load Baseline
    baseline = BaselineLogisticRegression.load(baseline_path)
    print(f"[2/3] Loaded Baseline Logistic Regression from: {baseline_path}")

    # Load World Model System
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
        num_classes=num_classes,
    )
    system = NetworkWorldModelSystem(gnn_encoder, world_model, threat_head).to(device)
    system.load_state_dict(checkpoint["model_state_dict"])
    system.eval()
    print(f"      Loaded World Model System from: {world_model_path}")

    rollout_engine = AutoregressiveRolloutEngine(
        world_model=world_model,
        threat_head=threat_head,
        horizon_k=horizon_k,
        sequence_length=seq_len,
    )

    # 2. Multi-Step Evaluation Loop
    print(f"\n[3/3] Evaluating horizons t+1 ... t+{horizon_k} on TEST partition ...")

    # Pre-encode all test graphs into latent vectors z
    with torch.no_grad():
        test_z_list = []
        for g in test_graphs:
            g_dev = g.to(device)
            z = system.encode_graph(g_dev)
            test_z_list.append(z.squeeze(0))
        test_z_tensor = torch.stack(test_z_list, dim=0)  # [T, 128]

    # Containers for ground truth and predictions per horizon step k (k=1..K)
    y_true_by_k = {k: [] for k in range(1, horizon_k + 1)}
    y_pred_wm_by_k = {k: [] for k in range(1, horizon_k + 1)}
    y_pred_base_by_k = {k: [] for k in range(1, horizon_k + 1)}

    num_test_steps = len(test_graphs) - seq_len - horizon_k
    for i in range(num_test_steps):
        # Historical context: z[t-N+1:t]
        z_history = test_z_tensor[i : i + seq_len]  # [N, 128]
        # Current window flat features for baseline
        x_flat_curr = test_graphs[i + seq_len - 1].flat_features.detach().cpu().numpy().reshape(1, -1)

        # Autoregressive K-step rollout for World Model
        wm_trajectory = rollout_engine.rollout(z_history)

        # Baseline prediction
        base_pred = int(baseline.predict(x_flat_curr)[0])

        for step_record in wm_trajectory:
            k = step_record["step"]
            actual_future_graph = test_graphs[i + seq_len + k - 1]
            actual_y = int(actual_future_graph.y.item())

            y_true_by_k[k].append(actual_y)
            y_pred_wm_by_k[k].append(step_record["predicted_class_index"])
            # Baseline static projection across horizon
            y_pred_base_by_k[k].append(base_pred)

    # 3. Compute Metrics & Compile Comparison Report
    horizon_results = []
    print("\n" + "=" * 90)
    print(f"{'Horizon':<8} {'Lead Time':<12} {'World Model Macro F1':<24} {'Baseline Macro F1':<20} {'Advantage':>10}")
    print("=" * 90)

    for k in range(1, horizon_k + 1):
        lead_info = compute_lead_time_gain(k, window_seconds=win_sec)
        wm_metrics = compute_evaluation_metrics(y_true_by_k[k], y_pred_wm_by_k[k], class_names)
        base_metrics = compute_evaluation_metrics(y_true_by_k[k], y_pred_base_by_k[k], class_names)

        diff = wm_metrics["macro_f1"] - base_metrics["macro_f1"]
        sign = "+" if diff >= 0 else ""

        print(
            f"t+{k:<6} {lead_info['lead_time_seconds']}s ({lead_info['lead_time_minutes']}m)      "
            f"{wm_metrics['macro_f1']:.4f} (FPR: {wm_metrics['false_positive_rate']:.3f})     "
            f"{base_metrics['macro_f1']:.4f} (FPR: {base_metrics['false_positive_rate']:.3f})     "
            f"{sign}{diff:.4f}"
        )

        horizon_results.append({
            "horizon_step": k,
            "horizon_label": f"t+{k}",
            "lead_time": lead_info,
            "world_model": wm_metrics,
            "baseline": base_metrics,
            "f1_improvement": round(float(diff), 4),
        })

    # Save to JSON
    report_data = {
        "evaluation_metadata": {
            "project": "SIH26153",
            "test_windows_count": len(test_graphs),
            "sequence_length": seq_len,
            "forecast_horizon_k": horizon_k,
            "window_seconds": win_sec,
        },
        "comparative_summary": horizon_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("=" * 90)
    print(f"[✓] Full evaluation report written to: {output_path.resolve()}")
    print("=" * 90)
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)

if __name__ == "__main__":
    main()
