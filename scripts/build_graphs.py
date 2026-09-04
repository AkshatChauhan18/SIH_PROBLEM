#!/usr/bin/env python3
"""
scripts/build_graphs.py

Constructs PyTorch Geometric Data graphs S[t] for all temporal windows.
Applies:
- Priority edge sampling guardrail (MAX_EDGES = 10,000 for RTX 4060 VRAM safety)
- Robust feature scaling (fitted on TRAIN graphs only — Zero Leakage)
- Graph serialization to data/graphs/{train,val,test}_graphs.pt
"""

import sys
import argparse
import pickle
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.preprocessing.normalization import PreprocessorRegistry
from src.graph.features import extract_node_and_edge_features
from src.graph.builder import build_pyg_graph_from_window

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Construct PyG graphs S[t] from window records.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = Path(config["paths"]["processed_dir"])
    graphs_dir = Path(config["paths"]["graphs_dir"])
    graphs_dir.mkdir(parents=True, exist_ok=True)
    preproc_path = Path(config["paths"]["preprocessors_path"])

    max_edges = config["graph_construction"]["max_edges"]
    self_loops = config["graph_construction"]["add_self_loops"]
    node_feature_dim = config["graph_construction"]["node_feature_dim"]
    edge_feature_dim = config["graph_construction"]["edge_feature_dim"]
    # flat_feature_dim matches number of stats computed in windowing.py (always 8)
    flat_feature_dim = config["graph_construction"].get("flat_feature_dim", 8)

    print("=" * 80)
    print("SIH 2026: GRAPH CONSTRUCTION & SCALING PIPELINE")
    print("=" * 80)
    print(f"Processed Directory: {processed_dir.resolve()}")
    print(f"Graphs Output Dir  : {graphs_dir.resolve()}")
    print(f"MAX_EDGES Guardrail: {max_edges:,}\n")

    preprocessors = PreprocessorRegistry.load(preproc_path)

    # 1. Load window partitions
    print("[1/4] Loading window records ...")
    with open(processed_dir / "train_windows.pkl", "rb") as f:
        train_wins = pickle.load(f)
    with open(processed_dir / "val_windows.pkl", "rb") as f:
        val_wins = pickle.load(f)
    with open(processed_dir / "test_windows.pkl", "rb") as f:
        test_wins = pickle.load(f)
    print(f"      Train: {len(train_wins)} | Val: {len(val_wins)} | Test: {len(test_wins)}")

    # 2. Fit RobustScalers on TRAIN graph features only (Zero Leakage)
    print("\n[2/4] Extracting and fitting graph feature scalers on TRAIN windows ...")
    train_node_feats_list = []
    train_edge_feats_list = []
    for w in train_wins:
        _, n_f, _, e_f = extract_node_and_edge_features(w["flows"])
        if len(n_f) > 0:
            train_node_feats_list.append(n_f)
        if len(e_f) > 0:
            train_edge_feats_list.append(e_f)

    stacked_node_feats = np.vstack(train_node_feats_list) if train_node_feats_list else np.empty((0, node_feature_dim))
    stacked_edge_feats = np.vstack(train_edge_feats_list) if train_edge_feats_list else np.empty((0, edge_feature_dim))

    preprocessors.fit_graph_scalers(stacked_node_feats, stacked_edge_feats)
    preprocessors.save(preproc_path)
    print(f"      Fitted node scaler on {len(stacked_node_feats):,} nodes, edge scaler on {len(stacked_edge_feats):,} edges.")

    # 3. Build PyG Data graphs for all splits
    def convert_split(windows, split_name):
        print(f"  [+] Building PyG graphs for {split_name} ({len(windows)} windows) ... ", end="", flush=True)
        graphs: List[Data] = []
        for w in windows:
            g = build_pyg_graph_from_window(
                w,
                preprocessors=preprocessors,
                max_edges=max_edges,
                add_self_loops=self_loops,
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                flat_feature_dim=flat_feature_dim,
            )
            graphs.append(g)
        print("Done.")
        return graphs

    print("\n[3/4] Building scaled PyG graphs ...")
    train_graphs = convert_split(train_wins, "TRAIN")
    val_graphs = convert_split(val_wins, "VALIDATION")
    test_graphs = convert_split(test_wins, "TEST")

    # 4. Serialize graphs to disk
    print("\n[4/4] Serializing graph collections ...")
    torch.save(train_graphs, graphs_dir / "train_graphs.pt")
    torch.save(val_graphs, graphs_dir / "val_graphs.pt")
    torch.save(test_graphs, graphs_dir / "test_graphs.pt")

    print("\n" + "=" * 80)
    print(f"[✓] Graphs built and saved successfully to: {graphs_dir.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
