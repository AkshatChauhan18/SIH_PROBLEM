#!/usr/bin/env python3
"""
scripts/preprocess.py

Preprocesses raw CIC-IDS2017 CSV files into temporal windows using a
PER-FILE chronological 70 / 15 / 15 split strategy.

Why per-file (not global) chronological splitting:
- CIC-IDS2017 schedules specific attack types on specific days of the week.
- A global chronological split would put BOTNET (Friday) exclusively in TEST,
  meaning the model NEVER sees BOTNET during training and cannot predict it.
- Per-file splitting ensures that every attack type in every capture file is
  proportionally represented across Train / Val / Test, while still
  maintaining strict temporal ordering WITHIN each file (no flow leakage).

ZERO DATA LEAKAGE GUARANTEES:
- Normalization statistics (median imputation) fitted on TRAIN flows only.
- Feature scalers (RobustScaler) fitted on TRAIN graphs only.
- Each file is split chronologically: first 70% → Train, next 15% → Val, last 15% → Test.
- Sequences in train_world_model.py never cross partition boundaries.
"""

import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
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


def print_class_distribution(name: str, windows: List[Dict[str, Any]], class_names: List[str]) -> None:
    """Prints formatted class distribution table for a split partition."""
    total = len(windows)
    counts = Counter(w["target_category"] for w in windows)
    print(f"\n  {name} ({total:,} windows):")
    print(f"    {'Category':<24} {'Count':>7} {'Percentage':>12}")
    print("    " + "-" * 45)
    for cat in class_names:
        cnt = counts.get(cat, 0)
        pct = (cnt / total) * 100 if total > 0 else 0.0
        print(f"    {cat:<24} {cnt:>7} {pct:>11.2f}%")


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
    split_config = config.get("split", {})
    train_ratio = float(split_config.get("train_ratio", 0.70))
    val_ratio   = float(split_config.get("val_ratio",   0.15))

    class_names = config["threat_head"]["classes"]

    print("=" * 80)
    print("SIH 2026: PER-FILE CHRONOLOGICAL TEMPORAL WINDOW PREPROCESSING")
    print("=" * 80)
    print(f"Raw Data Directory : {raw_dir.resolve()}")
    print(f"Window Size        : {window_sec} seconds")
    print(f"Split Strategy     : Per-file chronological {train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {(1-train_ratio-val_ratio)*100:.0f}%")
    print(f"Output Directory   : {processed_dir.resolve()}")
    print(f"\nRationale: Per-file splitting ensures attack types that occur on specific")
    print(f"days (e.g. BOTNET on Friday) are proportionally represented in all splits.\n")

    adapter = CICIDS2017Adapter()
    all_files = sorted(list(raw_dir.glob("*.csv")))
    print(f"Found {len(all_files)} raw CSV files:")
    for f in all_files:
        print(f"  - {f.name}")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # Pass 1: Ingest the first 70% of each file to fit the training medians
    #         (zero-leakage: scalers are never exposed to val/test flows)
    # ──────────────────────────────────────────────────────────────────────────
    print("[1/4] Fitting median imputation statistics on TRAIN flow slices (first 70% of each file) ...")
    preprocessors = PreprocessorRegistry()
    train_frames = []
    loaded_dfs: Dict[str, pl.DataFrame] = {}

    for fp in all_files:
        print(f"  [+] Ingesting: {fp.name} ... ", end="", flush=True)
        lf = adapter.load_flows(fp)
        df = lf.collect()
        loaded_dfs[fp.name] = df
        n_train_flows = int(train_ratio * len(df))
        train_frames.append(df.slice(0, n_train_flows))
        print(f"Done ({len(df):,} flows, {n_train_flows:,} train flows)")

    merged_train_df = pl.concat(train_frames)
    preprocessors.fit_flow_medians(merged_train_df)
    preproc_path = Path(config["paths"]["preprocessors_path"])
    preprocessors.save(preproc_path)
    print(f"  [✓] Fitted {len(preprocessors.flow_medians)} feature medians on {len(merged_train_df):,} training flows.")
    print(f"  [✓] Saved preprocessors to: {preproc_path.resolve()}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Pass 2: Per-file chronological window construction + 70/15/15 split
    # ──────────────────────────────────────────────────────────────────────────
    print("[2/4] Constructing temporal windows with per-file 70/15/15 chronological split ...")
    train_windows: List[Dict[str, Any]] = []
    val_windows:   List[Dict[str, Any]] = []
    test_windows:  List[Dict[str, Any]] = []

    for fp in all_files:
        df = loaded_dfs[fp.name]
        df_clean = clean_and_impute_flows(df, medians=preprocessors.flow_medians)

        # partition_flows_into_windows already sorts internally by timestamp
        windows = partition_flows_into_windows(df_clean, window_seconds=window_sec)
        n_win = len(windows)
        n_tr  = int(train_ratio * n_win)
        n_vl  = int(val_ratio   * n_win)

        file_tr = windows[:n_tr]
        file_vl = windows[n_tr : n_tr + n_vl]
        file_ts = windows[n_tr + n_vl :]

        train_windows.extend(file_tr)
        val_windows.extend(file_vl)
        test_windows.extend(file_ts)

        # Count attack classes per file split to show the distribution
        tr_cats  = Counter(w["target_category"] for w in file_tr)
        val_cats = Counter(w["target_category"] for w in file_vl)
        ts_cats  = Counter(w["target_category"] for w in file_ts)
        attack_tr  = {k: v for k, v in tr_cats.items()  if k != "BENIGN"}
        attack_val = {k: v for k, v in val_cats.items() if k != "BENIGN"}
        attack_ts  = {k: v for k, v in ts_cats.items()  if k != "BENIGN"}
        print(f"  [✓] {fp.name}:")
        print(f"       {n_win} windows → {len(file_tr)} Train | {len(file_vl)} Val | {len(file_ts)} Test")
        if attack_tr or attack_val or attack_ts:
            print(f"       Attacks — Train: {attack_tr} | Val: {attack_val} | Test: {attack_ts}")

    # Re-index window_idx sequentially within each partition
    for i, w in enumerate(train_windows):
        w["window_idx"] = i
    for i, w in enumerate(val_windows):
        w["window_idx"] = i
    for i, w in enumerate(test_windows):
        w["window_idx"] = i

    print()

    # ──────────────────────────────────────────────────────────────────────────
    # Pass 3: Serialize partitions to disk
    # ──────────────────────────────────────────────────────────────────────────
    print("[3/4] Serializing partitioned window records ...")
    with open(processed_dir / "train_windows.pkl", "wb") as f:
        pickle.dump(train_windows, f)
    with open(processed_dir / "val_windows.pkl", "wb") as f:
        pickle.dump(val_windows, f)
    with open(processed_dir / "test_windows.pkl", "wb") as f:
        pickle.dump(test_windows, f)
    print(f"  [✓] Saved to: {processed_dir.resolve()}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Verification Report
    # ──────────────────────────────────────────────────────────────────────────
    total_wins = len(train_windows) + len(val_windows) + len(test_windows)
    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"  Total Windows : {total_wins:,}")
    print(f"  Train Windows : {len(train_windows):,} ({len(train_windows)/total_wins*100:.2f}%)")
    print(f"  Val Windows   : {len(val_windows):,} ({len(val_windows)/total_wins*100:.2f}%)")
    print(f"  Test Windows  : {len(test_windows):,} ({len(test_windows)/total_wins*100:.2f}%)")

    print("\n[4/4] Class distributions across splits:")
    print_class_distribution("TRAIN",      train_windows, class_names)
    print_class_distribution("VALIDATION", val_windows,   class_names)
    print_class_distribution("TEST",       test_windows,  class_names)

    # Temporal boundary spot-check (best-effort: sort train/val/test by start_time)
    train_sorted = sorted(train_windows, key=lambda w: w["start_time"])
    val_sorted   = sorted(val_windows,   key=lambda w: w["start_time"])
    test_sorted  = sorted(test_windows,  key=lambda w: w["start_time"])
    print("\n" + "=" * 80)
    print("TEMPORAL SPAN PER PARTITION (across all files)")
    print("=" * 80)
    print(f"  Train : {train_sorted[0]['start_time']}  →  {train_sorted[-1]['end_time']}")
    print(f"  Val   : {val_sorted[0]['start_time']}  →  {val_sorted[-1]['end_time']}")
    print(f"  Test  : {test_sorted[0]['start_time']}  →  {test_sorted[-1]['end_time']}")
    print()

    print("=" * 80)
    print("[✓] Preprocessing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
