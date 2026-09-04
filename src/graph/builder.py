"""
src/graph/builder.py
Constructs PyTorch Geometric Data objects S[t] from temporal flow windows.
Integrates priority sampling guardrail and pre-fitted feature scalers.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from torch_geometric.data import Data
import polars as pl

from src.graph.features import extract_node_and_edge_features
from src.graph.sampling import apply_priority_edge_sampling
from src.preprocessing.normalization import PreprocessorRegistry

def build_pyg_graph_from_window(
    window_record: Dict[str, Any],
    preprocessors: Optional[PreprocessorRegistry] = None,
    max_edges: int = 10000,
    add_self_loops: bool = False,
    node_feature_dim: int = 9,
    edge_feature_dim: int = 7,
    flat_feature_dim: int = 8,
) -> Data:
    """
    Constructs a PyTorch Geometric Data graph S[t] from a temporal window record.

    Target labels are NEVER included as node/edge features.
    """
    flows_df: pl.DataFrame = window_record["flows"]
    target_idx: int = window_record["target_index"]
    target_cat: str = window_record["target_category"]
    w_idx: int = window_record["window_idx"]
    flat_feat: list = window_record.get("flat_features", [])

    nodes, node_feats, edge_pairs, edge_feats = extract_node_and_edge_features(flows_df)

    # Apply priority edge sampling guardrail (RTX 4060 VRAM safety)
    if len(edge_pairs) > max_edges:
        edge_pairs, edge_feats = apply_priority_edge_sampling(
            edge_pairs, edge_feats, max_edges=max_edges
        )

    # Normalize features using pre-fitted training scalers
    if preprocessors and preprocessors.is_fitted:
        node_feats = preprocessors.transform_node_features(node_feats)
        edge_feats = preprocessors.transform_edge_features(edge_feats)

    node_to_idx = {ip: i for i, ip in enumerate(nodes)}
    num_nodes = len(nodes)

    if num_nodes == 0:
        # Handle rare completely empty window
        x = torch.zeros((1, node_feature_dim), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_feature_dim), dtype=torch.float32)
        node_map = ["dummy"]
    else:
        x = torch.tensor(node_feats, dtype=torch.float32)
        if len(edge_pairs) > 0:
            src_indices = [node_to_idx[u] for u, v in edge_pairs]
            dst_indices = [node_to_idx[v] for u, v in edge_pairs]
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_feature_dim), dtype=torch.float32)
        node_map = nodes

    # Optional self-loops
    if add_self_loops and num_nodes > 0:
        self_loop_idx = torch.arange(num_nodes, dtype=torch.long)
        loop_edges = torch.stack([self_loop_idx, self_loop_idx], dim=0)
        edge_index = torch.cat([edge_index, loop_edges], dim=1)
        zero_attr = torch.zeros((num_nodes, edge_attr.size(1) if edge_attr.size(1) > 0 else edge_feature_dim), dtype=torch.float32)
        edge_attr = torch.cat([edge_attr, zero_attr], dim=0)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([target_idx], dtype=torch.long),
        num_nodes=num_nodes if num_nodes > 0 else 1,
    )
    # Metadata attributes
    pyg_data.window_idx = w_idx
    pyg_data.target_category = target_cat
    pyg_data.node_ips = node_map
    pyg_data.flat_features = torch.tensor(flat_feat, dtype=torch.float32) if flat_feat else torch.zeros(flat_feature_dim)
    pyg_data.start_time_iso = window_record["start_time"].isoformat() if hasattr(window_record["start_time"], "isoformat") else str(window_record["start_time"])

    return pyg_data
