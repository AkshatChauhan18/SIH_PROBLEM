#!/usr/bin/env python3
"""
scripts/train_baseline.py

Trains Logistic Regression baseline using flat temporal-window aggregated statistics.
Forecasts future attack/event state y[t+1] using strictly past information.
Evaluates on validation set and saves model to models/baseline_model.pkl.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.metrics import compute_evaluation_metrics
from src.models.baseline import BaselineLogisticRegression
from src.preprocessing.normalization import CLASS_INDEX_TO_NAME

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def extract_flat_sequences(graphs):
    """
    Extracts flat features at time t and future target at time t+1:
    X[i] = flat_features[t]
    y[i] = target[t+1]
    """
    X, y = [], []
    for t in range(len(graphs) - 1):
        x_t = graphs[t].flat_features.numpy()
        y_next = int(graphs[t + 1].y.item())
        X.append(x_t)
        y.append(y_next)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    parser = argparse.ArgumentParser(description="Train baseline Logistic Regression on window statistics.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    graphs_dir = Path(config["paths"]["graphs_dir"])
    model_path = Path(config["paths"]["baseline_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_names = config["threat_head"]["classes"]

    print("=" * 80)
    print("SIH 2026: BASELINE LOGISTIC REGRESSION TRAINING")
    print("=" * 80)
    print(f"Target Model Path: {model_path.resolve()}\n")

    train_graphs = torch.load(graphs_dir / "train_graphs.pt", weights_only=False)
    val_graphs = torch.load(graphs_dir / "val_graphs.pt", weights_only=False)

    X_train, y_train = extract_flat_sequences(train_graphs)
    X_val, y_val = extract_flat_sequences(val_graphs)

    print(f"Train Samples: {len(X_train)} (Features: {X_train.shape[1]})")
    print(f"Val Samples  : {len(X_val)}")

    baseline = BaselineLogisticRegression(random_state=config["project"]["random_seed"])
    print("\nFitting Logistic Regression with balanced class weights ...")
    baseline.fit(X_train, y_train)

    # Evaluate on Validation set
    y_pred_val = baseline.predict(X_val)
    val_metrics = compute_evaluation_metrics(y_val, y_pred_val, class_names)

    print("\n" + "-" * 80)
    print(f"Validation Macro F1        : {val_metrics['macro_f1']:.4f}")
    print(f"Validation Weighted F1     : {val_metrics['weighted_f1']:.4f}")
    print(f"Validation Macro Precision : {val_metrics['macro_precision']:.4f}")
    print(f"Validation Macro Recall    : {val_metrics['macro_recall']:.4f}")
    print(f"Validation False Pos Rate  : {val_metrics['false_positive_rate']:.4f}")
    print("-" * 80)

    print("\nPer-Class Performance:")
    for name, stats in val_metrics["per_class"].items():
        if stats["support"] > 0:
            print(f"  {name:<22} F1: {stats['f1']:.4f} | Prec: {stats['precision']:.4f} | Rec: {stats['recall']:.4f} | Support: {stats['support']}")

    baseline.save(model_path)
    print(f"\n[✓] Baseline model saved to: {model_path.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
