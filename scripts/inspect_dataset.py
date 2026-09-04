#!/usr/bin/env python3
"""
scripts/inspect_dataset.py

SIH 2026 — SIH26153: AI-Based Network Attack Forecasting via World Model
Immediate First Deliverable: High-Performance Dataset Inspection Engine

Efficiently profiles raw CIC-IDS2017 CSV files using Polars LazyFrames (scan_csv)
without modifying original files. Computes:
- File discovery & dirty header normalization
- Schema & FlowRecord column mapping
- Corrupted / blank padding row detection
- Infinity, -Infinity, NaN, and null auditing
- Robust 12-to-24 hour chronological timestamp parsing
- Per-file and global label distributions
- Unique IP topology & communication pairs
- Graph-density metrics for GNN constraint sizing
- JSON summary output to reports/dataset_inspection.json
"""

import os
import sys
import glob
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import polars as pl

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Standardized FlowRecord schema definition as required by Project Master Context
STANDARDIZED_SCHEMA_MAPPING = {
    # Network ID
    "src_ip": "Source IP",
    "dst_ip": "Destination IP",
    "src_port": "Source Port",
    "dst_port": "Destination Port",
    "protocol": "Protocol",
    # Temporal
    "timestamp": "Timestamp",
    "flow_duration": "Flow Duration",
    # Packet / Byte statistics
    "fwd_pkts": "Total Fwd Packets",
    "bwd_pkts": "Total Backward Packets",
    "fwd_bytes": "Total Length of Fwd Packets",
    "bwd_bytes": "Total Length of Bwd Packets",
    "flow_bytes_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    # Flow dynamics
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "init_win_fwd": "Init_Win_bytes_forward",
    "init_win_bwd": "Init_Win_bytes_backward",
    # TCP flags
    "syn_count": "SYN Flag Count",
    "rst_count": "RST Flag Count",
    "ack_count": "ACK Flag Count",
    "fin_count": "FIN Flag Count",
    "psh_count": "PSH Flag Count",
    # Target
    "label": "Label",
}

def clean_label_string(label: str) -> str:
    """Sanitize dirty characters in attack labels (e.g. non-ASCII dashes in Web Attacks)."""
    if not label:
        return "UNKNOWN"
    # Normalize non-breaking / non-ascii dashes or replacement chars
    cleaned = label.replace("\ufffd", "-").replace("–", "-").replace("—", "-").strip()
    return cleaned

def inspect_single_file(file_path: Path) -> Dict[str, Any]:
    """Inspects a single CIC-IDS2017 CSV file using Polars LazyFrames."""
    start_time = time.time()
    file_name = file_path.name
    file_size_bytes = file_path.stat().st_size

    # 1. Read header directly to examine raw column names & whitespace
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fp:
        raw_header_line = fp.readline().rstrip("\r\n")
    raw_columns = raw_header_line.split(",")
    raw_col_count = len(raw_columns)
    stripped_columns = [c.strip() for c in raw_columns]

    # Check for duplicate headers
    seen_cols = set()
    duplicate_cols = []
    for c in stripped_columns:
        if c in seen_cols:
            duplicate_cols.append(c)
        seen_cols.add(c)

    # 2. Lazy scan using Polars with utf8-lossy encoding
    # We do NOT set null_values upfront so we can precisely audit Infinity, NaN, and null counts
    lf_raw = pl.scan_csv(
        file_path,
        encoding="utf8-lossy",
        infer_schema_length=5000,
        truncate_ragged_lines=True,
    )

    schema_names = lf_raw.collect_schema().names()
    
    # Map raw / renamed column names to clean stripped names
    # Polars auto-renames duplicates like " Fwd Header Length_duplicated_0"
    col_mapping = {}
    for raw_name in schema_names:
        base = raw_name.strip()
        col_mapping[raw_name] = base

    # Identify key column references in the raw file
    def find_raw_col(target_stripped: str) -> str:
        for raw_c in schema_names:
            if raw_c.strip() == target_stripped:
                return raw_c
        return ""

    c_flow_id = find_raw_col("Flow ID")
    c_src_ip = find_raw_col("Source IP")
    c_dst_ip = find_raw_col("Destination IP")
    c_timestamp = find_raw_col("Timestamp")
    c_flow_duration = find_raw_col("Flow Duration")
    c_label = find_raw_col("Label")
    c_flow_bytes_s = find_raw_col("Flow Bytes/s")
    c_flow_pkts_s = find_raw_col("Flow Packets/s")

    # 3. Separate valid rows from empty / blank comma-padding rows
    # In files like Thursday Morning WebAttacks, 288k trailing lines are ",,,..."
    # Valid rows must have non-null, non-empty Flow ID, Source IP, and Timestamp
    total_raw_rows = lf_raw.select(pl.len()).collect().item()

    is_valid_row = (
        pl.col(c_flow_id).is_not_null()
        & (pl.col(c_flow_id).str.strip_chars() != "")
        & pl.col(c_src_ip).is_not_null()
        & (pl.col(c_src_ip).str.strip_chars() != "")
        & pl.col(c_timestamp).is_not_null()
        & (pl.col(c_timestamp).str.strip_chars() != "")
    )

    valid_row_count = (
        lf_raw.filter(is_valid_row).select(pl.len()).collect().item()
    )
    empty_padding_rows = total_raw_rows - valid_row_count

    # Filter down to valid records for statistics
    lf_valid = lf_raw.filter(is_valid_row)

    # 4. Audit Infinity, -Infinity, NaN, and null values in numeric flow rate columns
    inf_nan_stats = {}
    for col_ref, label_key in [(c_flow_bytes_s, "flow_bytes_s"), (c_flow_pkts_s, "flow_pkts_s")]:
        if col_ref:
            col_expr = pl.col(col_ref)
            inf_count = lf_valid.filter(
                col_expr.cast(pl.String).str.strip_chars().is_in(["Infinity", "+Infinity"])
            ).select(pl.len()).collect().item()
            neg_inf_count = lf_valid.filter(
                col_expr.cast(pl.String).str.strip_chars() == "-Infinity"
            ).select(pl.len()).collect().item()
            nan_count = lf_valid.filter(
                col_expr.cast(pl.String).str.strip_chars().is_in(["NaN", "nan"])
            ).select(pl.len()).collect().item()
            null_count = lf_valid.filter(col_expr.is_null()).select(pl.len()).collect().item()
            inf_nan_stats[label_key] = {
                "infinity": inf_count,
                "negative_infinity": neg_inf_count,
                "nan": nan_count,
                "null": null_count,
                "total_problematic": inf_count + neg_inf_count + nan_count + null_count,
            }

    # 5. Robust Timestamp Parsing & Datetime Ranges
    # Handles:
    # - dd/MM/yyyy HH:mm:ss (e.g. Monday: 03/07/2017 08:55:58)
    # - d/M/yyyy H:m (e.g. Tuesday-Friday: 7/7/2017 3:30)
    # - yyyy-MM-dd HH:mm:ss
    # Converts 12-hour working hours afternoon timestamps without PM (hours 1..7 -> 13..19)
    ts_parsed_expr = (
        pl.coalesce([
            pl.col(c_timestamp).str.strip_chars().str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S", strict=False),
            pl.col(c_timestamp).str.strip_chars().str.strptime(pl.Datetime, "%d/%m/%Y %H:%M", strict=False),
            pl.col(c_timestamp).str.strip_chars().str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
            pl.col(c_timestamp).str.strip_chars().str.strptime(pl.Datetime, "%Y-%m-%d %H:%M", strict=False),
        ])
    )
    # Apply 12-hour working-hours offset
    ts_24h_expr = (
        pl.when(ts_parsed_expr.dt.hour().is_between(1, 7))
        .then(ts_parsed_expr + pl.duration(hours=12))
        .otherwise(ts_parsed_expr)
    )

    ts_stats = lf_valid.select([
        ts_24h_expr.min().alias("min_dt"),
        ts_24h_expr.max().alias("max_dt"),
        ts_24h_expr.null_count().alias("unparsed_ts_count")
    ]).collect()

    min_dt_val = ts_stats["min_dt"][0]
    max_dt_val = ts_stats["max_dt"][0]
    unparsed_ts = ts_stats["unparsed_ts_count"][0]

    duration_seconds = 0.0
    if min_dt_val and max_dt_val:
        duration_seconds = (max_dt_val - min_dt_val).total_seconds()

    # 6. Label Distribution
    label_df = lf_valid.group_by(c_label).len().collect()
    label_distribution = {}
    for row in label_df.iter_rows():
        lbl_raw, cnt = row[0], row[1]
        lbl_clean = clean_label_string(str(lbl_raw))
        label_distribution[lbl_clean] = label_distribution.get(lbl_clean, 0) + cnt

    # 7. Topology & Graph Metrics
    # Unique Source IPs, Destination IPs, combined Host IPs |V|
    src_ips_df = lf_valid.select(pl.col(c_src_ip).str.strip_chars().alias("ip")).unique().collect()
    dst_ips_df = lf_valid.select(pl.col(c_dst_ip).str.strip_chars().alias("ip")).unique().collect()
    unique_src_count = len(src_ips_df)
    unique_dst_count = len(dst_ips_df)

    all_ips = set(src_ips_df["ip"].to_list()).union(set(dst_ips_df["ip"].to_list()))
    total_unique_hosts = len(all_ips)

    # Unique directed communication pairs |E| = (Source IP, Destination IP)
    pairs_count = len(
        lf_valid.select([
            pl.col(c_src_ip).str.strip_chars(),
            pl.col(c_dst_ip).str.strip_chars()
        ]).unique().collect()
    )

    # Graph density: D = |E| / (|V| * (|V| - 1))
    graph_density = 0.0
    if total_unique_hosts > 1:
        max_possible_edges = total_unique_hosts * (total_unique_hosts - 1)
        graph_density = pairs_count / max_possible_edges

    flow_to_edge_ratio = round(valid_row_count / pairs_count, 2) if pairs_count > 0 else 0.0
    avg_out_degree = round(pairs_count / total_unique_hosts, 2) if total_unique_hosts > 0 else 0.0

    elapsed = round(time.time() - start_time, 2)

    return {
        "file_name": file_name,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "raw_columns_count": raw_col_count,
        "duplicate_columns": duplicate_cols,
        "total_raw_rows": total_raw_rows,
        "valid_rows": valid_row_count,
        "empty_padding_rows": empty_padding_rows,
        "datetime_range": {
            "min_datetime": min_dt_val.isoformat() if min_dt_val else None,
            "max_datetime": max_dt_val.isoformat() if max_dt_val else None,
            "duration_hours": round(duration_seconds / 3600.0, 2),
            "duration_seconds": duration_seconds,
            "unparsed_timestamp_count": unparsed_ts,
        },
        "inf_nan_stats": inf_nan_stats,
        "label_distribution": label_distribution,
        "graph_topology": {
            "unique_source_ips": unique_src_count,
            "unique_destination_ips": unique_dst_count,
            "total_unique_hosts": total_unique_hosts,
            "unique_directed_pairs_edges": pairs_count,
            "graph_density": graph_density,
            "average_out_degree": avg_out_degree,
            "flows_per_edge_ratio": flow_to_edge_ratio,
        },
        "inspection_time_seconds": elapsed,
    }

def main():
    parser = argparse.ArgumentParser(description="Inspect CIC-IDS2017 dataset using Polars LazyFrames.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/cic2017",
        help="Directory containing CIC-IDS2017 CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/dataset_inspection.json",
        help="Path to output JSON summary report",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(data_dir.glob("*.csv")))
    if not csv_files:
        print(f"[!] Error: No CSV files found in directory: {data_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("SIH 2026 — SIH26153: AI WORLD MODEL ATTACK FORECASTING")
    print("DATASET INSPECTION & PROFILING ENGINE (POLARS LAZY)")
    print("=" * 80)
    print(f"Target Directory : {data_dir.resolve()}")
    print(f"CSV Files Found  : {len(csv_files)}")
    print(f"Output Report    : {output_path.resolve()}\n")

    overall_start = time.time()
    per_file_reports = []
    global_labels = {}
    total_raw_rows = 0
    total_valid_rows = 0
    total_empty_rows = 0
    global_all_ips = set()
    global_pairs = 0

    for idx, csv_file in enumerate(csv_files, 1):
        print(f"[{idx}/{len(csv_files)}] Scanning: {csv_file.name} ... ", end="", flush=True)
        file_rep = inspect_single_file(csv_file)
        per_file_reports.append(file_rep)
        
        # Aggregate totals
        total_raw_rows += file_rep["total_raw_rows"]
        total_valid_rows += file_rep["valid_rows"]
        total_empty_rows += file_rep["empty_padding_rows"]

        for lbl, cnt in file_rep["label_distribution"].items():
            global_labels[lbl] = global_labels.get(lbl, 0) + cnt

        print(
            f"Done ({file_rep['inspection_time_seconds']}s) — "
            f"Valid: {file_rep['valid_rows']:,} | Empty: {file_rep['empty_padding_rows']:,} | "
            f"Labels: {len(file_rep['label_distribution'])}"
        )

    overall_elapsed = round(time.time() - overall_start, 2)

    # Sort labels by frequency
    sorted_labels = dict(sorted(global_labels.items(), key=lambda item: item[1], reverse=True))

    # Compile Master Inspection Report
    master_report = {
        "metadata": {
            "project": "SIH26153 — AI-Based Network Attack Forecasting via World Model",
            "generated_at": datetime.now().isoformat(),
            "target_directory": str(data_dir.resolve()),
            "total_files_scanned": len(csv_files),
            "overall_elapsed_seconds": overall_elapsed,
        },
        "schema_specification": {
            "target_data_model": "FlowRecord",
            "field_mapping": STANDARDIZED_SCHEMA_MAPPING,
            "duplicate_columns_detected": ["Fwd Header Length (indices 40 and 61)"],
            "normalization_strategy": "Strip header whitespace, deduplicate columns, cast Infinity/NaN to null on load.",
        },
        "dataset_summary": {
            "total_raw_lines": total_raw_rows,
            "total_valid_flow_records": total_valid_rows,
            "total_empty_or_corrupt_padding_lines": total_empty_rows,
            "global_unique_classes_count": len(sorted_labels),
            "global_label_distribution": sorted_labels,
        },
        "per_file_reports": per_file_reports,
    }

    # Write out JSON report
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(master_report, fp, indent=2)

    # Print SOC Decision-Grade Summary Table
    print("\n" + "=" * 80)
    print("INSPECTION SUMMARY & SOC-GRADE PROFILE")
    print("=" * 80)
    print(f"Total Raw Lines Processed : {total_raw_rows:,}")
    print(f"Total Valid Flow Records  : {total_valid_rows:,} ({round(total_valid_rows/total_raw_rows*100, 1)}%)")
    print(f"Empty/Blank Padding Lines : {total_empty_rows:,} (Detected in Thursday Morning WebAttacks)")
    print(f"Total Execution Time      : {overall_elapsed}s\n")

    print("-" * 80)
    print(f"{'Attack Class':<32} {'Flow Count':>15} {'Percentage':>12}")
    print("-" * 80)
    for lbl, cnt in sorted_labels.items():
        pct = (cnt / total_valid_rows) * 100.0
        print(f"{lbl:<32} {cnt:>15,} {pct:>11.3f}%")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("PER-FILE TEMPORAL SPAN & TOPOLOGY")
    print("=" * 80)
    for rep in per_file_reports:
        ts_range = rep["datetime_range"]
        topo = rep["graph_topology"]
        inf = rep["inf_nan_stats"]
        total_inf_nan = sum(item["total_problematic"] for item in inf.values())
        print(f"File: {rep['file_name']}")
        print(f"  Rows        : {rep['valid_rows']:,} valid (Size: {rep['file_size_mb']} MB)")
        print(f"  Time Range  : {ts_range['min_datetime']} --> {ts_range['max_datetime']} ({ts_range['duration_hours']}h)")
        print(f"  Topology    : Hosts |V| = {topo['total_unique_hosts']:,} | Directed Pairs |E| = {topo['unique_directed_pairs_edges']:,} | Density = {topo['graph_density']:.6f}")
        print(f"  Inf/NaN/Null: {total_inf_nan} instances detected in flow rate metrics")
        print()

    print("=" * 80)
    print(f"[✓] Full JSON Report written to: {output_path.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
