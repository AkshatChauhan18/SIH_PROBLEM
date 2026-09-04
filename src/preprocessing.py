"""
Temporal State Preprocessing for SENTINEL-X
Aggregates packet-derived flow telemetry into time-binned network state vectors.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from .config import (
    STATE_FEATURES,
    LABEL_TO_STAGE,
    STAGE_TO_ID,
    STAGE_NAMES
)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names, drops duplicates, and handles infinite/NaN values."""
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    
    # Drop duplicate column names if any (e.g. Fwd Header Length.1)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Replace inf and -inf with NaN, then fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    
    return df

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parses mixed datetime strings and sorts by timestamp."""
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df

def aggregate_temporal_windows(df: pd.DataFrame, window_size_seconds: int = 5) -> pd.DataFrame:
    """
    Aggregates individual network flows into fixed temporal windows.
    Extracts high-dimensional network state vectors.
    """
    df = clean_dataframe(df)
    df = parse_timestamps(df)
    
    if len(df) == 0:
        return pd.DataFrame()
        
    # Map raw labels to canonical stages and binary attack indicator
    label_col = "Label" if "Label" in df.columns else None
    if label_col:
        df["mapped_stage"] = df[label_col].map(lambda l: LABEL_TO_STAGE.get(str(l).strip(), "Normal"))
        df["is_attack"] = df[label_col].apply(lambda l: 0 if str(l).strip() == "BENIGN" else 1)
    else:
        df["mapped_stage"] = "Normal"
        df["is_attack"] = 0

    # Ensure required columns exist with safe defaults
    req_numeric = {
        "Total Fwd Packets": 1,
        "Total Backward Packets": 0,
        "Total Length of Fwd Packets": 64,
        "Total Length of Bwd Packets": 0,
        "SYN Flag Count": 0,
        "ACK Flag Count": 0,
        "RST Flag Count": 0,
        "FIN Flag Count": 0,
        "PSH Flag Count": 0,
        "Flow IAT Mean": 0.0,
        "Flow IAT Std": 0.0,
        "Flow IAT Max": 0.0,
        "Flow IAT Min": 0.0,
        "Packet Length Mean": 64.0,
        "Packet Length Std": 0.0,
        "Packet Length Variance": 0.0,
        "Average Packet Size": 64.0,
        "Down/Up Ratio": 0.0,
        "Active Mean": 0.0,
        "Idle Mean": 0.0,
        "Init_Win_bytes_forward": 0,
        "Init_Win_bytes_backward": 0,
    }
    for col, default_val in req_numeric.items():
        if col not in df.columns:
            df[col] = default_val

    # Set timestamp index for resampling
    df["dt"] = df["Timestamp"]
    df = df.set_index("dt")

    # Time-window grouping rule
    freq_str = f"{window_size_seconds}s"
    resampled = df.resample(freq_str)

    state_rows = []
    
    for window_time, group in resampled:
        n_flows = len(group)
        if n_flows == 0:
            continue
            
        fwd_pkts = group["Total Fwd Packets"].sum()
        bwd_pkts = group["Total Backward Packets"].sum()
        tot_pkts = max(int(fwd_pkts + bwd_pkts), 1)
        
        fwd_bytes = group["Total Length of Fwd Packets"].sum()
        bwd_bytes = group["Total Length of Bwd Packets"].sum()
        tot_bytes = max(float(fwd_bytes + bwd_bytes), 0.0)
        
        u_src = group["Source IP"].nunique() if "Source IP" in group.columns else 1
        u_dst = group["Destination IP"].nunique() if "Destination IP" in group.columns else 1
        u_dst_ports = group["Destination Port"].nunique() if "Destination Port" in group.columns else 1
        
        port_diversity = float(u_dst_ports) / max(n_flows, 1)
        dest_diversity = float(u_dst) / max(n_flows, 1)
        
        syn_ratio = float(group["SYN Flag Count"].sum()) / tot_pkts
        ack_ratio = float(group["ACK Flag Count"].sum()) / tot_pkts
        rst_ratio = float(group["RST Flag Count"].sum()) / tot_pkts
        fin_ratio = float(group["FIN Flag Count"].sum()) / tot_pkts
        psh_ratio = float(group["PSH Flag Count"].sum()) / tot_pkts
        
        flow_pkts_sec = float(tot_pkts) / window_size_seconds
        flow_bytes_sec = float(tot_bytes) / window_size_seconds
        
        attack_ratio = float(group["is_attack"].mean())
        
        # Determine dominant stage
        stage_counts = group["mapped_stage"].value_counts()
        # If attack ratio > 0.05, prioritize the attack stage over Normal
        attack_stages = stage_counts[stage_counts.index != "Normal"]
        if not attack_stages.empty and attack_ratio > 0.05:
            dominant_stage = attack_stages.index[0]
        else:
            dominant_stage = stage_counts.index[0]
            
        stage_id = STAGE_TO_ID.get(dominant_stage, 0)
        
        row = {
            "window_timestamp": window_time,
            "total_flows": float(n_flows),
            "total_packets": float(tot_pkts),
            "total_bytes": float(tot_bytes),
            "unique_source_ips": float(u_src),
            "unique_dest_ips": float(u_dst),
            "unique_dest_ports": float(u_dst_ports),
            "port_diversity": port_diversity,
            "dest_diversity": dest_diversity,
            "syn_ratio": syn_ratio,
            "ack_ratio": ack_ratio,
            "rst_ratio": rst_ratio,
            "fin_ratio": fin_ratio,
            "psh_ratio": psh_ratio,
            "flow_packets_per_sec": flow_pkts_sec,
            "flow_bytes_per_sec": flow_bytes_sec,
            "flow_iat_mean": float(group["Flow IAT Mean"].mean()),
            "flow_iat_std": float(group["Flow IAT Std"].mean()),
            "flow_iat_max": float(group["Flow IAT Max"].max()),
            "flow_iat_min": float(group["Flow IAT Min"].min()),
            "packet_length_mean": float(group["Packet Length Mean"].mean()),
            "packet_length_std": float(group["Packet Length Std"].mean()),
            "packet_length_variance": float(group["Packet Length Variance"].mean()),
            "average_packet_size": float(group["Average Packet Size"].mean()),
            "down_up_ratio": float(group["Down/Up Ratio"].mean()),
            "active_mean": float(group["Active Mean"].mean()),
            "idle_mean": float(group["Idle Mean"].mean()),
            "init_window_forward_mean": float(group["Init_Win_bytes_forward"].mean()),
            "init_window_backward_mean": float(group["Init_Win_bytes_backward"].mean()),
            # Ground truth targets (NOT features for the model)
            "attack_ratio": attack_ratio,
            "binary_attack": 1 if (attack_ratio > 0.01 or dominant_stage != "Normal") else 0,
            "dominant_stage": dominant_stage,
            "stage_id": stage_id
        }
        state_rows.append(row)
        
    return pd.DataFrame(state_rows)

def create_sequences(
    state_df: pd.DataFrame, 
    seq_len: int = 10, 
    horizon: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs temporal sequence datasets:
    X: shape (N, seq_len, num_features)
    y_next_state: shape (N, num_features) - S(t+1)
    y_attack: shape (N, horizon) - binary attack indicator for next 1..K steps
    y_stage: shape (N, horizon) - stage IDs for next 1..K steps
    """
    feature_matrix = state_df[STATE_FEATURES].values
    attack_labels = state_df["binary_attack"].values
    stage_labels = state_df["stage_id"].values
    
    total_len = len(state_df)
    required_len = seq_len + horizon
    
    if total_len < required_len:
        return (np.empty((0, seq_len, len(STATE_FEATURES))), 
                np.empty((0, len(STATE_FEATURES))),
                np.empty((0, horizon)), 
                np.empty((0, horizon)))
        
    X_list = []
    y_next_state_list = []
    y_attack_list = []
    y_stage_list = []
    
    for i in range(total_len - required_len + 1):
        x_seq = feature_matrix[i : i + seq_len]
        y_next_state = feature_matrix[i + seq_len]
        y_attack_horizon = attack_labels[i + seq_len : i + seq_len + horizon]
        y_stage_horizon = stage_labels[i + seq_len : i + seq_len + horizon]
        
        X_list.append(x_seq)
        y_next_state_list.append(y_next_state)
        y_attack_list.append(y_attack_horizon)
        y_stage_list.append(y_stage_horizon)
        
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_next_state_list, dtype=np.float32),
        np.array(y_attack_list, dtype=np.float32),
        np.array(y_stage_list, dtype=np.int64)
    )
