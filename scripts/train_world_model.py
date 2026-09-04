#!/usr/bin/env python3
"""
scripts/train_world_model.py

Trains GNN Encoder + GRU World Model + ThreatHead end-to-end using Dual Loss:
Loss = MSE(pred_z[t+1], actual_z[t+1]) + lambda * CrossEntropy(ThreatHead(pred_z[t+1]), target_event[t+1])

Features:
- Fixed latent vector dimension z[t] = 128 (64 mean pool + 64 max pool)
- GRU learns latent dynamics P(z[t+1] | z[t-N+1:t])
- ThreatHead evaluates predicted future latent vector against future event ground truth
- Mixed precision training with torch.amp for NVIDIA RTX 4060
- Gradient clipping and checkpoint saving to models/world_model.pt
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.device import get_device, set_seed, get_autocast_context
from src.utils.metrics import compute_evaluation_metrics
from src.models.gnn_encoder import GNNEncoder
from src.models.prediction_head import ThreatHead
from src.models.world_model import WorldModel, NetworkWorldModelSystem

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

class TemporalGraphSequenceDataset(Dataset):
    """
    Constructs chronological sequences of N graph states predicting the (N+1)-th state:
    Sequence: [S[t-N+1], ..., S[t]] -> Target: S[t+1], y[t+1]
    """

    def __init__(self, graphs: List, seq_len: int = 10):
        self.graphs = graphs
        self.seq_len = seq_len
        self.valid_indices = range(len(graphs) - seq_len)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        seq_graphs = [self.graphs[idx + i] for i in range(self.seq_len)]
        target_graph = self.graphs[idx + self.seq_len]
        return seq_graphs, target_graph

def collate_temporal_batch(batch):
    """Custom collator preserving sequence structure of PyG graph windows."""
    seq_list, target_list = [], []
    for seq_graphs, target_g in batch:
        seq_list.append(seq_graphs)
        target_list.append(target_g)
    return seq_list, target_list

def main():
    parser = argparse.ArgumentParser(description="Train World Model on temporal graph sequences.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs in config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size in config")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["random_seed"])

    device = get_device(config["training"]["device"])
    use_amp = config["training"]["use_amp"] and (device.type == "cuda")

    graphs_dir = Path(config["paths"]["graphs_dir"])
    model_path = Path(config["paths"]["world_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    seq_len = config["temporal_windowing"]["sequence_length"]
    window_sec = config["temporal_windowing"]["window_seconds"]
    horizon_k = config["temporal_windowing"]["forecast_horizon"]
    epochs = args.epochs or config["training"]["epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    lambda_threat = float(config["training"]["lambda_threat"])
    grad_clip = float(config["training"]["grad_clip"])
    class_names = config["threat_head"]["classes"]
    num_classes = len(class_names)

    print("=" * 80)
    print("SIH 2026: GNN ENCODER + GRU WORLD MODEL TRAINING")
    print("=" * 80)
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Mixed Precision    : {use_amp}")
    print(f"Sequence Length N  : {seq_len} windows ({seq_len * window_sec}s historical context)")
    print(f"Dual Loss Lambda   : {lambda_threat} (MSE + lambda * CE)")
    print(f"Model Checkpoint   : {model_path.resolve()}\n")

    # 1. Load Precomputed PyG Graphs
    print("[1/4] Loading graph collections ...")
    train_graphs = torch.load(graphs_dir / "train_graphs.pt", weights_only=False)
    val_graphs = torch.load(graphs_dir / "val_graphs.pt", weights_only=False)
    print(f"      Train Graphs: {len(train_graphs):,} | Val Graphs: {len(val_graphs):,}")

    train_dataset = TemporalGraphSequenceDataset(train_graphs, seq_len=seq_len)
    val_dataset = TemporalGraphSequenceDataset(val_graphs, seq_len=seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_temporal_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_batch
    )
    print(f"      Train Sequences: {len(train_dataset):,} | Val Sequences: {len(val_dataset):,}")

    # Compute balanced class weights to mitigate attack imbalance across all 7 classes
    print("\n[2/4] Initializing World Model architecture ...")
    train_targets = [int(g.y.item()) for g in train_graphs]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    total_samples = len(train_targets)
    weights = total_samples / (num_classes * np.maximum(class_counts, 1.0))
    weights[0] = 0.20   # Downweight majority BENIGN so all attacks are actively prioritized
    weights = np.clip(weights, 0.20, 5.0)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"      Fitted Class Weights: {dict(zip(class_names, [round(float(w), 3) for w in weights]))}")

    # Initialize model components
    gnn_encoder = GNNEncoder(
        node_dim=config["graph_construction"]["node_feature_dim"],
        edge_dim=config["graph_construction"]["edge_feature_dim"],
        hidden_dim=config["model_architecture"]["gnn_hidden_dim"],
        dropout=config["model_architecture"]["dropout"],
    )
    world_model = WorldModel(
        latent_dim=config["model_architecture"]["latent_dim"],
        gru_hidden_dim=config["model_architecture"]["gru_hidden_dim"],
        gru_num_layers=config["model_architecture"]["gru_num_layers"],
        dropout=config["model_architecture"]["dropout"],
    )
    threat_head = ThreatHead(
        latent_dim=config["model_architecture"]["latent_dim"],
        hidden_dim=config["threat_head"]["hidden_dim"],
        num_classes=num_classes,
        dropout=config["model_architecture"]["dropout"],
    )

    system = NetworkWorldModelSystem(
        gnn_encoder=gnn_encoder,
        world_model=world_model,
        threat_head=threat_head,
        lambda_threat=lambda_threat,
        class_weights=class_weights_tensor,
    ).to(device)

    optimizer = torch.optim.AdamW(system.parameters(), lr=lr, weight_decay=weight_decay)
    use_bf16 = use_amp and (device.type == "cuda") and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and not use_bf16))

    print(f"      GNN Encoder Parameters : {sum(p.numel() for p in gnn_encoder.parameters()):,}")
    print(f"      World Model Parameters : {sum(p.numel() for p in world_model.parameters()):,}")
    print(f"      Threat Head Parameters : {sum(p.numel() for p in threat_head.parameters()):,}")
    print(f"      Total Trainable Params : {sum(p.numel() for p in system.parameters() if p.requires_grad):,}")

    # 3. Training Loop
    print("\n[3/4] Starting training epochs ...")
    best_train_loss = float("inf")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        system.train()
        train_total_loss = 0.0
        train_mse_loss = 0.0
        train_ce_loss = 0.0
        epoch_start = time.time()

        for seq_batch, target_batch in train_loader:
            optimizer.zero_grad()

            with get_autocast_context(device, use_amp):
                # Vectorized batch encoding: flatten all B * N graphs into a single PyG Batch
                flat_seq_graphs = [g for seq in seq_batch for g in seq]
                batch_graphs = Batch.from_data_list(flat_seq_graphs).to(device)
                z_all = system.encode_graph(batch_graphs)  # [B * N, 128]
                z_history = z_all.view(len(seq_batch), seq_len, -1)  # [B, N, 128]

                # Vectorized target graph encoding: B target graphs in one GPU call
                target_batch_graphs = Batch.from_data_list(target_batch).to(device)
                actual_z_next = system.encode_graph(target_batch_graphs)  # [B, 128]
                target_y_next = torch.tensor(
                    [int(g.y.item()) for g in target_batch], dtype=torch.long, device=device
                )

                # Predict next latent state via World Model
                pred_z_next, _ = system.world_model(z_history)  # [B, 128]

                # Compute Dual Loss
                loss, mse, ce = system.compute_dual_loss(
                    pred_z_next, actual_z_next, target_y_next
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(system.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_total_loss += loss.item()
            train_mse_loss += mse.item()
            train_ce_loss += ce.item()

        avg_train_loss = train_total_loss / max(len(train_loader), 1)
        avg_train_mse = train_mse_loss / max(len(train_loader), 1)
        avg_train_ce = train_ce_loss / max(len(train_loader), 1)

        # Validation Phase
        system.eval()
        val_total_loss = 0.0
        val_y_true = []
        val_y_pred = []

        with torch.no_grad():
            for seq_batch, target_batch in val_loader:
                with get_autocast_context(device, use_amp):
                    flat_seq_graphs = [g for seq in seq_batch for g in seq]
                    batch_graphs = Batch.from_data_list(flat_seq_graphs).to(device)
                    z_all = system.encode_graph(batch_graphs)
                    z_history = z_all.view(len(seq_batch), seq_len, -1)

                    target_batch_graphs = Batch.from_data_list(target_batch).to(device)
                    actual_z_next = system.encode_graph(target_batch_graphs)
                    target_y_next = torch.tensor(
                        [int(g.y.item()) for g in target_batch], dtype=torch.long, device=device
                    )

                    pred_z_next, _ = system.world_model(z_history)
                    v_loss, _, _ = system.compute_dual_loss(pred_z_next, actual_z_next, target_y_next)

                    logits = system.threat_head(pred_z_next)
                    preds = torch.argmax(logits, dim=-1)

                    val_total_loss += v_loss.item()
                    val_y_true.extend(target_y_next.cpu().numpy().tolist())
                    val_y_pred.extend(preds.cpu().numpy().tolist())

        avg_val_loss = val_total_loss / max(len(val_loader), 1)
        val_metrics = compute_evaluation_metrics(val_y_true, val_y_pred, class_names)
        elapsed = round(time.time() - epoch_start, 1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed}s) — "
            f"Train Loss: {avg_train_loss:.4f} (MSE: {avg_train_mse:.4f}, CE: {avg_train_ce:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f} | FPR: {val_metrics['false_positive_rate']:.4f}"
        )

        # Save best model checkpoint based on training convergence
        # (Avoids discarding trained epochs due to Thursday unseen-class validation shifts)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": system.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                model_path,
            )

    # Always ensure final epoch checkpoint is written if not already saved
    if not model_path.is_file():
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": system.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_metrics": val_metrics,
                "config": config,
            },
            model_path,
        )

    print("\n" + "=" * 80)
    print(f"[✓] Training Complete! Best model checkpoint saved to: {model_path.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
