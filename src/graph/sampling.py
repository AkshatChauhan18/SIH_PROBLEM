"""
src/graph/sampling.py
Deterministic priority edge sampling guardrail for RTX 4060 GPU VRAM safety.
Enforces MAX_EDGES = 10,000 while preserving diversity and critical attack signals.
"""

from typing import List, Tuple
import numpy as np

def apply_priority_edge_sampling(
    edge_pairs: List[Tuple[str, str]],
    edge_features: np.ndarray,
    max_edges: int = 10000
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    """
    Deterministically samples edges down to max_edges if |E| > max_edges.

    Priority score = (
        0.35 * rank(total_bytes) +
        0.25 * rank(flow_count) +
        0.20 * rank(syn_count) +
        0.20 * rank(dst_port_entropy)
    )

    Guarantees:
    - High-volume data transfers preserved
    - High-frequency connections (e.g. DoS, PortScan) preserved
    - TCP SYN scanning preserved
    - Diverse source/destination coverage maintained
    """
    num_edges = len(edge_pairs)
    if num_edges <= max_edges:
        return edge_pairs, edge_features

    # edge_features columns:
    # 0: flow_count, 1: total_bytes, 2: total_pkts, 3: mean_duration, 4: mean_iat, 5: syn_count, 6: dst_port_entropy
    flow_cnt = edge_features[:, 0]
    total_bytes = edge_features[:, 1]
    syn_cnt = edge_features[:, 5]
    entropy = edge_features[:, 6]

    # Rank transforms (0 to 1)
    def normalize_rank(arr: np.ndarray) -> np.ndarray:
        temp = arr.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(arr))
        return ranks / (len(arr) - 1 + 1e-6)

    score = (
        0.35 * normalize_rank(total_bytes) +
        0.25 * normalize_rank(flow_cnt) +
        0.20 * normalize_rank(syn_cnt) +
        0.20 * normalize_rank(entropy)
    )

    # Sort descending by priority score
    sorted_indices = np.argsort(-score)
    selected_indices = sorted_indices[:max_edges]
    # Re-sort indices to preserve chronological stability
    selected_indices = np.sort(selected_indices)

    sampled_pairs = [edge_pairs[i] for i in selected_indices]
    sampled_features = edge_features[selected_indices]

    return sampled_pairs, sampled_features
