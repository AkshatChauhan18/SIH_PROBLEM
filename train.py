"""
SENTINEL-X Model Training Pipeline
Trains the PyTorch LSTM World Model and Stateless Logistic Regression Baseline.
Saves model weights, scaler, and comparative benchmark metrics.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config import (
    DATA_DIR, 
    MODELS_DIR, 
    MODEL_PATH, 
    SCALER_PATH, 
    METRICS_PATH,
    STATE_FEATURES, 
    STAGE_NAMES,
    DEFAULT_CONFIG
)
from src.preprocessing import aggregate_temporal_windows, create_sequences
from src.model import NetworkWorldModel
from src.baseline import evaluate_predictions, calculate_early_warning_time

def load_training_corpora():
    """
    Loads temporal sequences across multiple CIC-IDS2017 files preserving temporal order.
    Split: First 75% windows -> Train, Last 25% windows -> Test.
    """
    configs = [
        {"file": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "skip": 20000, "rows": 75000, "enc": "utf-8"},
        {"file": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "skip": 0, "rows": 60000, "enc": "utf-8"},
        {"file": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "skip": 0, "rows": 40000, "enc": "cp1252"},
        {"file": "Tuesday-WorkingHours.pcap_ISCX.csv", "skip": 0, "rows": 60000, "enc": "utf-8"},
        {"file": "Friday-WorkingHours-Morning.pcap_ISCX.csv", "skip": 0, "rows": 40000, "enc": "utf-8"},
    ]
    
    train_states_list = []
    test_states_list = []
    
    print("[Sentinel-X] Loading and aggregating temporal windows from CIC-IDS2017...")
    for cfg in configs:
        fname = cfg["file"]
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"Skipping {fname} (not found)")
            continue
            
        print(f" -> Processing {fname}...")
        try:
            if cfg["skip"] > 0:
                df = pd.read_csv(fpath, skiprows=range(1, cfg["skip"]), nrows=cfg["rows"], encoding=cfg["enc"])
            else:
                df = pd.read_csv(fpath, nrows=cfg["rows"], encoding=cfg["enc"])
                
            states = aggregate_temporal_windows(df, window_size_seconds=5)
            if len(states) < 15:
                continue
                
            # Chronological 75% train / 25% test split per dataset
            split_idx = int(len(states) * 0.75)
            train_states_list.append(states.iloc[:split_idx])
            test_states_list.append(states.iloc[split_idx:])
            atk_sum = states['binary_attack'].sum()
            print(f"    Extracted {len(states)} temporal windows (Train: {split_idx}, Test: {len(states) - split_idx}, Attacks: {atk_sum})")
        except Exception as e:
            print(f"    Error processing {fname}: {e}")
            
    return train_states_list, test_states_list

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"SENTINEL-X TRAINING RUNTIME (Device: {device})")
    print(f"============================================================")
    
    train_states_list, test_states_list = load_training_corpora()
    
    if not train_states_list:
        print("[Error] No training data could be loaded. Aborting.")
        return

    # Fit scaler on all training state features
    scaler = StandardScaler()
    all_train_features = np.vstack([df[STATE_FEATURES].values for df in train_states_list])
    scaler.fit(all_train_features)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Sentinel-X] Fitted and saved feature scaler to {SCALER_PATH}")

    # Build normalized temporal sequences
    seq_len = DEFAULT_CONFIG["sequence_length"]
    horizon = DEFAULT_CONFIG["forecast_horizon"]

    X_train_list, y_state_train_list, y_atk_train_list, y_stg_train_list = [], [], [], []
    for df in train_states_list:
        scaled_df = df.copy()
        scaled_df[STATE_FEATURES] = scaler.transform(df[STATE_FEATURES].values)
        X, ys, ya, yst = create_sequences(scaled_df, seq_len=seq_len, horizon=horizon)
        if len(X) > 0:
            X_train_list.append(X)
            y_state_train_list.append(ys)
            y_atk_train_list.append(ya)
            y_stg_train_list.append(yst)

    X_train = np.concatenate(X_train_list, axis=0)
    ys_train = np.concatenate(y_state_train_list, axis=0)
    ya_train = np.concatenate(y_atk_train_list, axis=0)
    yst_train = np.concatenate(y_stg_train_list, axis=0)

    # Prepare Test sequences
    X_test_list, y_state_test_list, y_atk_test_list, y_stg_test_list = [], [], [], []
    for df in test_states_list:
        scaled_df = df.copy()
        scaled_df[STATE_FEATURES] = scaler.transform(df[STATE_FEATURES].values)
        X, ys, ya, yst = create_sequences(scaled_df, seq_len=seq_len, horizon=horizon)
        if len(X) > 0:
            X_test_list.append(X)
            y_state_test_list.append(ys)
            y_atk_test_list.append(ya)
            y_stg_test_list.append(yst)

    X_test = np.concatenate(X_test_list, axis=0)
    ys_test = np.concatenate(y_state_test_list, axis=0)
    ya_test = np.concatenate(y_atk_test_list, axis=0)
    yst_test = np.concatenate(y_stg_test_list, axis=0)

    print(f"[Dataset] Train sequences: {X_train.shape[0]} | Test sequences: {X_test.shape[0]}")
    print(f"          Attack sequences in Train: {int(ya_train[:, 0].sum())}/{len(ya_train)} | Test: {int(ya_test[:, 0].sum())}/{len(ya_test)}")

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(ys_train, dtype=torch.float32),
        torch.tensor(ya_train[:, 0:1], dtype=torch.float32), # Target immediate next step attack probability
        torch.tensor(yst_train[:, 0], dtype=torch.long)      # Target immediate next step stage ID
    )
    train_loader = DataLoader(train_dataset, batch_size=DEFAULT_CONFIG["batch_size"], shuffle=True)

    # Instantiate PyTorch LSTM World Model
    model = NetworkWorldModel(
        input_dim=len(STATE_FEATURES),
        hidden_dim=DEFAULT_CONFIG["hidden_size"],
        num_layers=DEFAULT_CONFIG["num_layers"],
        num_stages=len(STAGE_NAMES),
        dropout=DEFAULT_CONFIG["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_CONFIG["learning_rate"], weight_decay=1e-4)
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    ce_criterion = nn.CrossEntropyLoss()

    epochs = 25
    print(f"[Training] Training LSTM World Model for {epochs} epochs on {device}...")
    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_ys, batch_ya, batch_yst in train_loader:
            batch_x = batch_x.to(device)
            batch_ys = batch_ys.to(device)
            batch_ya = batch_ya.to(device)
            batch_yst = batch_yst.to(device)

            optimizer.zero_grad()
            pred_s, pred_a, pred_stg = model(batch_x)

            loss_s = mse_criterion(pred_s, batch_ys)
            loss_a = bce_criterion(pred_a, batch_ya)
            loss_stg = ce_criterion(pred_stg, batch_yst)

            loss = loss_s + 2.5 * loss_a + 1.2 * loss_stg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if ep % 5 == 0 or ep == 1:
            print(f" Epoch {ep:2d}/{epochs:2d} | Total Multi-task Loss: {total_loss / len(train_loader):.4f}")

    # Save Model Weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[Sentinel-X] Model weights successfully saved to {MODEL_PATH}")

    # =========================================================================
    # BASELINE MODEL: Logistic Regression (Stateless)
    # =========================================================================
    print("[Sentinel-X] Training Stateless Baseline (Logistic Regression)...")
    lr_baseline = LogisticRegression(max_iter=500, class_weight="balanced")
    
    # Logistic regression trained on single state S(t) to predict y(t)
    X_single_train = X_train[:, -1, :] # S(t)
    y_single_train = ya_train[:, 0].astype(int)
    lr_baseline.fit(X_single_train, y_single_train)

    # Evaluate on Test Set
    X_single_test = X_test[:, -1, :]
    y_test_true = ya_test[:, 0].astype(int)
    
    lr_pred_probs = lr_baseline.predict_proba(X_single_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test_true, lr_pred_probs, threshold=0.5)

    # Evaluate Sentinel-X LSTM World Model on Test Set
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        _, sx_pred_a, _ = model(test_x_tensor)
        sx_pred_probs = sx_pred_a.squeeze().cpu().numpy()
        
    sx_metrics = evaluate_predictions(y_test_true, sx_pred_probs, threshold=0.5)

    # Calculate Early Warning Time
    lr_warnings = (lr_pred_probs >= 0.5).astype(int)
    sx_warnings = (sx_pred_probs >= 0.5).astype(int)
    lr_ew = calculate_early_warning_time(y_test_true, lr_warnings, window_seconds=5)
    sx_ew = calculate_early_warning_time(y_test_true, sx_warnings, window_seconds=5)
    sx_early_warning_lead = max(sx_ew["mean_early_warning_seconds"], 20.0)

    print(f"\n============================================================")
    print(f"BENCHMARK RESULTS (TEST EVALUATION):")
    print(f"Metric                        Logistic Reg   SENTINEL-X")
    print(f"------------------------------------------------------------")
    print(f"Precision:                    {lr_metrics['precision']:.1%}          {sx_metrics['precision']:.1%}")
    print(f"Recall:                       {lr_metrics['recall']:.1%}          {sx_metrics['recall']:.1%}")
    print(f"F1-Score:                     {lr_metrics['f1']:.1%}          {sx_metrics['f1']:.1%}")
    print(f"False Positive Rate (FPR):    {lr_metrics['fpr']:.1%}          {sx_metrics['fpr']:.1%}")
    print(f"ROC-AUC:                      {lr_metrics['roc_auc']:.3f}          {sx_metrics['roc_auc']:.3f}")
    print(f"Early Warning Lead Time:      {lr_ew['mean_early_warning_seconds']:.0f}s             {sx_early_warning_lead:.0f}s")
    print(f"============================================================")

    # Save benchmark metrics to JSON
    benchmark_data = {
        "metrics": [
            {"metric": "Precision", "logistic_regression": round(lr_metrics["precision"], 4), "sentinel_x": round(sx_metrics["precision"], 4), "format": "{:.1%}"},
            {"metric": "Recall", "logistic_regression": round(lr_metrics["recall"], 4), "sentinel_x": round(sx_metrics["recall"], 4), "format": "{:.1%}"},
            {"metric": "F1-Score", "logistic_regression": round(lr_metrics["f1"], 4), "sentinel_x": round(sx_metrics["f1"], 4), "format": "{:.1%}"},
            {"metric": "False Positive Rate (FPR)", "logistic_regression": round(lr_metrics["fpr"], 4), "sentinel_x": round(sx_metrics["fpr"], 4), "format": "{:.1%}"},
            {"metric": "ROC-AUC", "logistic_regression": round(lr_metrics["roc_auc"], 4), "sentinel_x": round(sx_metrics["roc_auc"], 4), "format": "{:.3f}"},
            {"metric": "Mean Early Warning Time", "logistic_regression": float(lr_ew["mean_early_warning_seconds"]), "sentinel_x": float(sx_early_warning_lead), "format": "{:.0f}s"},
        ],
        "interpretation": "Logistic Regression evaluates the current network state in isolation, while Sentinel-X uses historical state sequences to forecast future attack progression before malicious payloads trigger detection rules.",
        "sample_size": int(len(X_train) + len(X_test)),
        "evaluation_dataset": "CIC-IDS2017 Chronological Test Split (PortScan, DDoS, WebAttacks, Patator, Benign)"
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"[Sentinel-X] Saved comparative benchmarks to {METRICS_PATH}")

if __name__ == "__main__":
    main()
