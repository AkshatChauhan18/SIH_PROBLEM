"""
src/graph/features.py
Node and edge statistical feature extraction for temporal network state graphs S[t].
Labels are strictly NEVER used as graph features.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import polars as pl

def compute_shannon_entropy(values: List[int]) -> float:
    """Computes Shannon entropy for port distribution."""
    if not values:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

def extract_node_and_edge_features(
    window_flows: pl.DataFrame,
) -> Tuple[List[str], np.ndarray, List[Tuple[str, str]], np.ndarray]:
    """
    Extracts topological and statistical features for nodes and edges within a window.

    Node features (9 dimensions):
    [in_degree, out_degree, in_bytes, out_bytes, in_pkts, out_pkts, syn_ratio, rst_ratio, peer_count]

    Edge features (7 dimensions):
    [flow_count, total_bytes, total_pkts, mean_duration, mean_iat, syn_count, dst_port_entropy]

    Returns:
    - node_list: list of unique IP strings
    - node_features: np.ndarray shape (|V|, 9)
    - edge_pairs: list of directed (src_ip, dst_ip) tuples
    - edge_features: np.ndarray shape (|E|, 7)
    """
    if len(window_flows) == 0:
        return [], np.empty((0, 9)), [], np.empty((0, 7))

    # All unique nodes in this window
    src_set = set(window_flows["src_ip"].to_list())
    dst_set = set(window_flows["dst_ip"].to_list())
    all_nodes = sorted(list(src_set.union(dst_set)))
    node_to_idx = {ip: i for i, ip in enumerate(all_nodes)}
    num_nodes = len(all_nodes)

    # 1. Edge Aggregation (Source IP -> Destination IP)
    # Aggregate multiple flows between the same host pair
    edge_agg = (
        window_flows.group_by(["src_ip", "dst_ip"])
        .agg([
            pl.len().alias("flow_count"),
            (pl.col("fwd_bytes").sum() + pl.col("bwd_bytes").sum()).alias("total_bytes"),
            (pl.col("fwd_pkts").sum() + pl.col("bwd_pkts").sum()).alias("total_pkts"),
            pl.col("flow_duration").mean().alias("mean_duration"),
            pl.col("flow_iat_mean").mean().alias("mean_iat"),
            pl.col("syn_count").sum().alias("syn_count"),
            pl.col("dst_port").alias("dst_ports"),
        ])
    )

    edge_pairs: List[Tuple[str, str]] = []
    edge_feats_list: List[List[float]] = []

    # Node accumulators
    in_degree = np.zeros(num_nodes, dtype=np.float32)
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    in_bytes = np.zeros(num_nodes, dtype=np.float32)
    out_bytes = np.zeros(num_nodes, dtype=np.float32)
    in_pkts = np.zeros(num_nodes, dtype=np.float32)
    out_pkts = np.zeros(num_nodes, dtype=np.float32)
    syn_accum = np.zeros(num_nodes, dtype=np.float32)
    rst_accum = np.zeros(num_nodes, dtype=np.float32)
    peers: Dict[int, set] = {i: set() for i in range(num_nodes)}

    for row in edge_agg.iter_rows(named=True):
        u, v = row["src_ip"], row["dst_ip"]
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]

        f_cnt = float(row["flow_count"])
        t_bytes = float(row["total_bytes"])
        t_pkts = float(row["total_pkts"])
        m_dur = float(row["mean_duration"] or 0.0)
        m_iat = float(row["mean_iat"] or 0.0)
        syn_c = float(row["syn_count"] or 0.0)
        ports = row["dst_ports"]
        entropy = compute_shannon_entropy(ports)

        edge_pairs.append((u, v))
        edge_feats_list.append([f_cnt, t_bytes, t_pkts, m_dur, m_iat, syn_c, entropy])

        # Accumulate node stats
        out_degree[u_idx] += 1.0
        in_degree[v_idx] += 1.0
        out_bytes[u_idx] += t_bytes
        in_bytes[v_idx] += t_bytes
        out_pkts[u_idx] += t_pkts
        in_pkts[v_idx] += t_pkts
        syn_accum[u_idx] += syn_c
        peers[u_idx].add(v_idx)
        peers[v_idx].add(u_idx)

    # Accumulate RST from raw flows for nodes
    rst_agg = window_flows.group_by("src_ip").agg(pl.col("rst_count").sum().alias("rst_sum"))
    for row in rst_agg.iter_rows(named=True):
        u = row["src_ip"]
        if u in node_to_idx:
            rst_accum[node_to_idx[u]] = float(row["rst_sum"] or 0.0)

    # Build node feature matrix
    peer_count = np.array([len(peers[i]) for i in range(num_nodes)], dtype=np.float32)
    total_node_pkts = out_pkts + in_pkts + 1e-6
    syn_ratio = syn_accum / total_node_pkts
    rst_ratio = rst_accum / total_node_pkts

    node_features = np.column_stack([
        in_degree,
        out_degree,
        in_bytes,
        out_bytes,
        in_pkts,
        out_pkts,
        syn_ratio,
        rst_ratio,
        peer_count,
    ]).astype(np.float32)

    edge_features = np.array(edge_feats_list, dtype=np.float32) if edge_feats_list else np.empty((0, 7), dtype=np.float32)

    return all_nodes, node_features, edge_pairs, edge_features
