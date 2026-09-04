"""
src/preprocessing/windowing.py
Chronological temporal window partitioner for network flow streams.
Constructs discrete temporal network states S[t].
"""

from typing import List, Dict, Any, Tuple
from datetime import timedelta
import polars as pl
from src.preprocessing.normalization import (
    SEVERITY_HIERARCHY,
    map_raw_label_to_category,
    map_category_to_index,
)

def assign_window_target_label(label_counts: Dict[str, int]) -> Tuple[str, int]:
    """
    Assigns target label to a temporal window using security severity hierarchy.
    Ensures subtle stealth attacks (e.g. Infiltration) are prioritized over concurrent benign traffic.
    """
    mapped_counts = {}
    for raw_lbl, cnt in label_counts.items():
        cat = map_raw_label_to_category(raw_lbl)
        mapped_counts[cat] = mapped_counts.get(cat, 0) + cnt

    # Check hierarchy
    for cat in SEVERITY_HIERARCHY:
        if mapped_counts.get(cat, 0) > 0:
            return cat, map_category_to_index(cat)

    return "BENIGN", 0

def partition_flows_into_windows(
    flows_df: pl.DataFrame,
    window_seconds: int = 30
) -> List[Dict[str, Any]]:
    """
    Partitions a sorted flow DataFrame into continuous temporal windows.
    Returns list of window records containing:
    - window_idx
    - start_time, end_time
    - flows_df (flows belonging strictly to this window)
    - target_category, target_index (ground truth event label)
    - window_summary_stats
    """
    if len(flows_df) == 0:
        return []

    # Ensure sorted by timestamp
    flows_df = flows_df.sort("timestamp")
    min_ts = flows_df["timestamp"][0]
    max_ts = flows_df["timestamp"][-1]

    # Calculate window ID relative to min_ts
    flows_with_win = flows_df.with_columns(
        ((pl.col("timestamp") - min_ts).dt.total_seconds() // window_seconds)
        .cast(pl.Int64)
        .alias("window_idx")
    )

    windows = []
    # Group by window_idx
    win_groups = flows_with_win.partition_by("window_idx", as_dict=True)

    sorted_indices = sorted(win_groups.keys())
    for w_key in sorted_indices:
        w_df = win_groups[w_key]
        raw_idx = w_key[0] if isinstance(w_key, tuple) else w_key
        w_idx_val = int(raw_idx)
        w_start = min_ts + timedelta(seconds=int(w_idx_val * window_seconds))
        w_end = w_start + timedelta(seconds=int(window_seconds))

        # Count labels in window
        lbl_counts_df = w_df.group_by("label").len()
        lbl_counts = dict(zip(lbl_counts_df["label"].to_list(), lbl_counts_df["len"].to_list()))
        target_cat, target_idx = assign_window_target_label(lbl_counts)

        # Flat statistical features for baseline model comparison
        total_flows = len(w_df)
        total_bytes = float(w_df["fwd_bytes"].sum() + w_df["bwd_bytes"].sum())
        total_pkts = float(w_df["fwd_pkts"].sum() + w_df["bwd_pkts"].sum())
        syn_ratio = float(w_df["syn_count"].sum()) / total_pkts if total_pkts > 0 else 0.0
        rst_ratio = float(w_df["rst_count"].sum()) / total_pkts if total_pkts > 0 else 0.0
        mean_duration = float(w_df["flow_duration"].mean() or 0.0)
        n_unique_src = w_df["src_ip"].n_unique()
        n_unique_dst = w_df["dst_ip"].n_unique()

        flat_features = [
            total_flows,
            total_bytes,
            total_pkts,
            syn_ratio,
            rst_ratio,
            mean_duration,
            n_unique_src,
            n_unique_dst,
        ]

        windows.append({
            "window_idx": int(w_idx_val),
            "start_time": w_start,
            "end_time": w_end,
            "flows": w_df,
            "target_category": target_cat,
            "target_index": target_idx,
            "flat_features": flat_features,
            "flow_count": total_flows,
        })

    return windows
