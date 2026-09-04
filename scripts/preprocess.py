#!/usr/bin/env python3
"""
scripts/preprocess.py

Preprocesses raw CIC-IDS2017 CSV files into continuous chronological temporal windows.
ZERO DATA LEAKAGE:
- Preprocessor statistics (median imputation) fitted on TRAIN (Days 1–3) only.
- Strict chronological splitting:
  * Train: Monday, Tuesday, Wednesday
  * Val: Thursday (Morning + Afternoon)
  * Test: Friday (Morning + Afternoon PortScan + Afternoon DDoS)
"""

import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any
import polars as pl

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.adapters.cic_ids2017 import CICIDS2017Adapter
from src.preprocessing.normalization import PreprocessorRegistry
from src.preprocessing.cleaning import clean_and_impute_flows
from src.preprocessing.windowing import partition_flows_into_windows

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def process_file_partition(
    file_paths: List[Path],
    adapter: CICIDS2017Adapter,
    medians: Dict[str, float] = None,
    window_seconds: int = 30,
) -> List[Dict[str, Any]]:
    """Loads and windows a set of CSV files."""
    all_windows = []
    for fp in sorted(file_paths):
        print(f"  [+] Loading flows: {fp.name} ... ", end="", flush=True)
        lf = adapter.load_flows(fp)
        df = lf.collect()
        df = clean_and_impute_flows(df, medians=medians)
        windows = partition_flows_into_windows(df, window_seconds=window_seconds)
        all_windows.extend(windows)
        print(f"Done ({len(df):,} flows -> {len(windows)} windows)")
    return all_windows

def main():
    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS2017 dataset into temporal windows.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw_data_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    window_sec = config["temporal_windowing"]["window_seconds"]
    splits = config["splits"]

    print("=" * 80)
    print("SIH 2026: TEMPORAL WINDOW PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"Raw Data Directory : {raw_dir.resolve()}")
    print(f"Window Size        : {window_sec} seconds")
    print(f"Output Directory   : {processed_dir.resolve()}\n")

    adapter = CICIDS2017Adapter()

    # Process all 8 CSV files chronologically (70% Train / 15% Val / 15% Test per file)
    all_files = sorted(list(raw_dir.glob("*.csv")))
    print(f"Total CSV Files ({len(all_files)}):")
    for f in all_files:
        print(f"  - {f.name}")
    print()

    # 1. Fit Preprocessor medians on TRAIN slices only (First 70% of each file — Zero Data Leakage)
    print("[1/4] Fitting median imputation statistics on TRAIN flow slices (First 70% of each day) ...")
    preprocessors = PreprocessorRegistry()
    train_frames = []
    loaded_dfs = {}

    for fp in all_files:
        print(f"  [+] Ingesting: {fp.name} ... ", end="", flush=True)
        lf = adapter.load_flows(fp)
        df = lf.collect()
        loaded_dfs[fp.name] = df
        # Extract first 70% of flows for zero-leakage training statistics
        n_train_flows = int(0.70 * len(df))
        train_frames.append(df.slice(0, n_train_flows))
        print(f"Done ({len(df):,} flows, {n_train_flows:,} train flows)")

    merged_train_df = pl.concat(train_frames)
    preprocessors.fit_flow_medians(merged_train_df)
    preproc_path = Path(config["paths"]["preprocessors_path"])
    preprocessors.save(preproc_path)
    print(f"      Fitted {len(preprocessors.flow_medians)} feature medians on {len(merged_train_df):,} flows.")
    print(f"      Saved preprocessors to: {preproc_path.resolve()}\n")

    # 2. Window each file and partition chronologically: 70% Train, 15% Val, 15% Test
    print("[2/4] Partitioning flows into 30s windows with chronological 70/15/15 split per file ...")
    train_windows = []
    val_windows = []
    test_windows = []

    for fp in all_files:
        df = loaded_dfs[fp.name]
        df_clean = clean_and_impute_flows(df, medians=preprocessors.flow_medians)
        windows = partition_flows_into_windows(df_clean, window_seconds=window_sec)
        
        n_win = len(windows)
        n_tr = int(0.70 * n_win)
        n_vl = int(0.15 * n_win)
        
        file_tr = windows[:n_tr]
        file_vl = windows[n_tr : n_tr + n_vl]
        file_ts = windows[n_tr + n_vl :]
        
        train_windows.extend(file_tr)
        val_windows.extend(file_vl)
        test_windows.extend(file_ts)
        print(f"  [✓] {fp.name}: {n_win} windows -> {len(file_tr)} Train | {len(file_vl)} Val | {len(file_ts)} Test")

    # Re-index window IDs sequentially within each partition
    for i, w in enumerate(train_windows):
        w["window_idx"] = i
    for i, w in enumerate(val_windows):
        w["window_idx"] = i
    for i, w in enumerate(test_windows):
        w["window_idx"] = i

    print(f"\n[3/4] Saving partitioned window records ...")
    print(f"      Total TRAIN windows: {len(train_windows):,}")
    print(f"      Total VAL windows  : {len(val_windows):,}")
    print(f"      Total TEST windows : {len(test_windows):,}")

    with open(processed_dir / "train_windows.pkl", "wb") as f:
        pickle.dump(train_windows, f)
    with open(processed_dir / "val_windows.pkl", "wb") as f:
        pickle.dump(val_windows, f)
    with open(processed_dir / "test_windows.pkl", "wb") as f:
        pickle.dump(test_windows, f)

    # Print class breakdown across splits
    from collections import Counter
    tr_counts = Counter(w["target_category"] for w in train_windows)
    vl_counts = Counter(w["target_category"] for w in val_windows)
    ts_counts = Counter(w["target_category"] for w in test_windows)
    print(f"\n[4/4] Partition Class Distributions:")
    print(f"      TRAIN: {dict(tr_counts)}")
    print(f"      VAL  : {dict(vl_counts)}")
    print(f"      TEST : {dict(ts_counts)}")

    print("\n" + "=" * 80)
    print(f"[✓] Preprocessing Complete! Datasets saved to: {processed_dir.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
