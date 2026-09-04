"""
Pre-extracted Real Temporal Scenarios for Demo Mode and Fast Walkthroughs.
Extracted directly from real CIC-IDS2017 flow data (PortScan, DDoS, WebAttacks, Benign).
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .config import DATA_DIR, STATE_FEATURES, LABEL_TO_STAGE
from .preprocessing import aggregate_temporal_windows, clean_dataframe

DEMO_CACHE_DIR = Path(__file__).resolve().parent.parent / "models" / "demo_cache"

def ensure_demo_cache(force: bool = False):
    """Generates cached real temporal sequences if not present."""
    DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    portscan_cache = DEMO_CACHE_DIR / "portscan_scenario.parquet"
    ddos_cache = DEMO_CACHE_DIR / "ddos_scenario.parquet"
    benign_cache = DEMO_CACHE_DIR / "benign_scenario.parquet"

    # Only extract if not already cached or force=True
    if force or not portscan_cache.exists():
        portscan_file = DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        if portscan_file.exists():
            try:
                # Capture transition from pre-attack benign baseline to active port scan
                df = pd.read_csv(portscan_file, skiprows=range(1, 25000), nrows=75000)
                states = aggregate_temporal_windows(df, window_size_seconds=5)
                if len(states) > 0:
                    states.to_parquet(portscan_cache)
                    print(f"[Sentinel-X] Cached real PortScan scenario ({len(states)} windows).")
            except Exception as e:
                print(f"[Warning] Failed caching PortScan: {e}")

    if force or not ddos_cache.exists():
        ddos_file = DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        if ddos_file.exists():
            try:
                df = pd.read_csv(ddos_file, nrows=60000)
                states = aggregate_temporal_windows(df, window_size_seconds=5)
                if len(states) > 0:
                    states.to_parquet(ddos_cache)
                    print(f"[Sentinel-X] Cached real DDoS scenario ({len(states)} windows).")
            except Exception as e:
                print(f"[Warning] Failed caching DDoS: {e}")

    if force or not benign_cache.exists():
        benign_file = DATA_DIR / "Friday-WorkingHours-Morning.pcap_ISCX.csv"
        if benign_file.exists():
            try:
                df = pd.read_csv(benign_file, nrows=40000)
                states = aggregate_temporal_windows(df, window_size_seconds=5)
                if len(states) > 0:
                    states.to_parquet(benign_cache)
                    print(f"[Sentinel-X] Cached real Benign scenario ({len(states)} windows).")
            except Exception as e:
                print(f"[Warning] Failed caching Benign: {e}")

def get_demo_scenarios() -> Dict[str, str]:
    """Returns available demonstration scenarios."""
    return {
        "Scenario 1: Reconnaissance to Infiltration (PortScan Transition)": "portscan",
        "Scenario 2: Traffic Surge to Denial of Service (DDoS Transition)": "ddos",
        "Scenario 3: Nominal Enterprise Baseline (Pure Benign Traffic)": "benign"
    }

def load_scenario_states(scenario_key: str = "portscan") -> pd.DataFrame:
    """Loads state trajectory dataframe for chosen scenario."""
    ensure_demo_cache()
    file_map = {
        "portscan": DEMO_CACHE_DIR / "portscan_scenario.parquet",
        "ddos": DEMO_CACHE_DIR / "ddos_scenario.parquet",
        "benign": DEMO_CACHE_DIR / "benign_scenario.parquet"
    }
    target_file = file_map.get(scenario_key, file_map["portscan"])
    if target_file.exists():
        return pd.read_parquet(target_file)
        
    # Fallback to direct extraction on demand
    csv_file = DATA_DIR / ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" if scenario_key == "portscan" else "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    if csv_file.exists():
        df = pd.read_csv(csv_file, nrows=30000)
        return aggregate_temporal_windows(df, window_size_seconds=5)
    return pd.DataFrame()

def load_sample_raw_flows(scenario_key: str = "portscan", n_rows: int = 150) -> pd.DataFrame:
    """Loads a sample of raw flows for the Traffic Analysis inspection table."""
    file_map = {
        "portscan": DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "ddos": DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "benign": DATA_DIR / "Friday-WorkingHours-Morning.pcap_ISCX.csv"
    }
    target_file = file_map.get(scenario_key, file_map["portscan"])
    if not target_file.exists():
        return pd.DataFrame()
        
    cols = [
        "Timestamp", "Source IP", "Destination IP", "Destination Port", 
        "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Label"
    ]
    try:
        # Read header and a sample of rows
        sample = pd.read_csv(target_file, nrows=2500)
        sample = clean_dataframe(sample)
        available = [c for c in cols if c in sample.columns]
        
        # Mix benign and attack flows if available
        attacks = sample[sample["Label"] != "BENIGN"]
        benign = sample[sample["Label"] == "BENIGN"]
        
        if len(attacks) > 0 and len(benign) > 0:
            combined = pd.concat([benign.head(n_rows // 2), attacks.head(n_rows // 2)])
        else:
            combined = sample.head(n_rows)
            
        combined = combined[available].copy()
        
        # Friendly renaming
        combined = combined.rename(columns={
            "Total Fwd Packets": "Fwd Packets",
            "Total Backward Packets": "Bwd Packets",
            "Total Length of Fwd Packets": "Fwd Bytes",
            "Total Length of Bwd Packets": "Bwd Bytes"
        })
        return combined
    except Exception as e:
        print(f"[Warning] Failed loading raw flows: {e}")
        return pd.DataFrame()
