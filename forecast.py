#!/usr/bin/env python
"""
SENTINEL-X Terminal Attack Forecaster
SIH Problem Statement SIH26153: AI based Network Attack Forecasting from Network Traffic Data
Tagline: "Don't just detect the attack. Forecast where it's going."

Executes forward trajectory simulation on a specified time window and compares
forecasted attack probabilities and MITRE stages against actual ground truth.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Optional

# Ensure UTF-8 output encoding if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    STATE_FEATURES,
    STAGE_NAMES,
    FEATURE_DISPLAY_NAMES,
    DEFAULT_CONFIG,
    DATA_DIR
)
from src.demo_data import load_scenario_states
from src.forecast_engine import ForecastEngine
from src.explainability import compute_feature_attributions
from src.preprocessing import aggregate_temporal_windows, clean_dataframe

def print_banner():
    banner = r"""
========================================================================================
   ____  _____ _   _ _____ ___ _   _ _____ _     __  __
  / ___|| ____| \ | |_   _|_ _| \ | | ____| |    \ \/ /
  \___ \|  _| |  \| | | |  | ||  \| |  _| | |     \  / 
   ___) | |___| |\  | | |  | || |\  | |___| |___  /  \ 
  |____/|_____|_| \_| |_| |___|_| \_|_____|_____|/_/\_\
  PREDICTIVE NETWORK ATTACK FORECASTING ENGINE  (SIH26153)
========================================================================================
"Don't just detect the attack. Forecast where it's going."
"""
    print(banner)

def list_available_windows(states_df: pd.DataFrame, max_rows: int = 30):
    print("\nAvailable Time Windows in Scenario:")
    print("-" * 75)
    print(f"{'Idx':>4} | {'Timestamp':<22} | {'Flows':>6} | {'Atk Ratio':>10} | {'Atk?':>5} | {'Dominant Stage':<16}")
    print("-" * 75)
    
    step = max(1, len(states_df) // max_rows)
    for idx in range(0, len(states_df), step):
        row = states_df.iloc[idx]
        ts_str = str(row.get('window_timestamp', f'Window {idx}'))
        flows = int(row.get('total_flows', 0))
        atk_r = row.get('attack_ratio', 0.0)
        is_atk = int(row.get('binary_attack', 0))
        stg = str(row.get('dominant_stage', 'Normal'))
        mark = "[!]" if is_atk else "[.]"
        print(f"{idx:>4} | {ts_str:<22} | {flows:>6} | {atk_r:>9.2%} | {mark} {is_atk} | {stg:<16}")
    print("-" * 75)
    print("Run: python forecast.py --window <Idx> to evaluate from that specific window.\n")

def run_cli():
    parser = argparse.ArgumentParser(
        description="Sentinel-X: Network Attack Trajectory Forecaster (Terminal CLI)"
    )
    parser.add_argument(
        "-s", "--scenario", 
        choices=["portscan", "ddos", "benign"], 
        default="portscan",
        help="Telemetry scenario (default: portscan)"
    )
    parser.add_argument(
        "-w", "--window", 
        type=int, 
        default=None,
        help="Time window index (t) to evaluate (default: auto-selects attack transition window)"
    )
    parser.add_argument(
        "-t", "--time",
        type=str,
        default=None,
        help="Search window by timestamp substring (e.g. '01:05')"
    )
    parser.add_argument(
        "-k", "--horizon", 
        type=int, 
        default=5,
        help="Forecast horizon steps lookahead (default: 5)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.50,
        help="Attack probability alert threshold (default: 0.50)"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Path to custom CIC-IDS2017 CSV file to process on demand"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available time windows in selected scenario"
    )

    args = parser.parse_args()
    print_banner()

    # 1. Load Data
    if args.file:
        if not os.path.exists(args.file):
            print(f"[Error] File not found: {args.file}")
            sys.exit(1)
        print(f"[*] Processing custom CSV: {args.file}...")
        df_raw = pd.read_csv(args.file, nrows=60000, encoding="cp1252" if "Web" in args.file else "utf-8")
        states_df = aggregate_temporal_windows(df_raw, window_size_seconds=5)
        scenario_name = os.path.basename(args.file)
    else:
        print(f"[*] Loading scenario: '{args.scenario.upper()}' (CIC-IDS2017 Telemetry)...")
        states_df = load_scenario_states(args.scenario)
        scenario_name = args.scenario.upper()

    if len(states_df) < 15:
        print(f"[Error] Insufficient temporal windows in dataset ({len(states_df)} found, need >= 15).")
        sys.exit(1)

    # If --list flag requested, show windows table and exit
    if args.list:
        list_available_windows(states_df)
        return

    # 2. Select Time Window (t)
    history_len = DEFAULT_CONFIG["sequence_length"]  # 10
    total_windows = len(states_df)
    max_eval_idx = total_windows - 1

    if args.time:
        matches = [i for i, ts in enumerate(states_df['window_timestamp']) if args.time in str(ts)]
        if matches:
            eval_idx = matches[0]
            print(f"[*] Found timestamp matching '{args.time}': Window {eval_idx}")
        else:
            print(f"[Warning] No window matching '{args.time}'. Defaulting to auto-selection.")
            eval_idx = None
    else:
        eval_idx = args.window

    # Auto-select an interesting attack progression window if not specified
    if eval_idx is None:
        if args.scenario == "portscan":
            # Window 18 is where PortScan escalates
            eval_idx = 18 if 18 < total_windows - args.horizon else 15
        elif args.scenario == "ddos":
            eval_idx = 25 if 25 < total_windows - args.horizon else 20
        else:
            eval_idx = 15

    # Clamp index safely
    eval_idx = max(history_len - 1, min(eval_idx, max_eval_idx))
    start_idx = eval_idx - history_len + 1
    
    # 3. Initialize Forecast Engine
    engine = ForecastEngine()
    device_name = "CUDA (RTX 4060)" if str(engine.device) == "cuda" else "CPU"
    model_mode = "PyTorch LSTM World Model" if engine.is_real_model else "Prototype Dynamics Mode"
    print(f"[*] AI Engine: {model_mode} on {device_name}")
    print(f"[*] Sequence History: Window {start_idx} to {eval_idx} ({history_len} temporal states, 50s duration)")
    print(f"[*] Target Observation Window S(t): Index {eval_idx} | Horizon K = {args.horizon} windows")
    print("=" * 88 + "\n")

    # Extract sequence
    history_slice = states_df.iloc[start_idx : eval_idx + 1]
    sequence_features = history_slice[STATE_FEATURES].values
    current_row = history_slice.iloc[-1]
    curr_timestamp = current_row.get("window_timestamp", f"Window {eval_idx}")

    # 4. Run Forecast Engine
    results = engine.forecast_trajectory(
        sequence_features=sequence_features,
        horizon=args.horizon,
        attack_threshold=args.threshold
    )

    # 5. Display Window Telemetry Card
    print("+--------------------------------------------------------------------------------------+")
    print("| CURRENT OBSERVATION WINDOW S(t) TELEMETRY                                            |")
    print("+--------------------------------------------------------------------------------------+")
    print(f"| Timestamp:            {str(curr_timestamp):<62} |")
    print(f"| Window Index:         t = {eval_idx:<58} |")
    print(f"| Flow Activity:        {int(current_row['total_flows']):<6} flows | {int(current_row['total_packets']):<8} pkts | {int(current_row['total_bytes']):<10} bytes                |")
    print(f"| Flow Rate:            {current_row['flow_packets_per_sec']:<8.1f} pkts/sec | {current_row['flow_bytes_per_sec']:<10.1f} bytes/sec                        |")
    print(f"| TCP Flag Ratios:      SYN: {current_row['syn_ratio']:<6.2%} | ACK: {current_row['ack_ratio']:<6.2%} | RST: {current_row['rst_ratio']:<6.2%} | PSH: {current_row['psh_ratio']:<6.2%}       |")
    print(f"| Topology Diversity:   Port Diversity: {current_row['port_diversity']:<6.3f} | Destination Diversity: {current_row['dest_diversity']:<6.3f}           |")
    print(f"| Flow IAT Jitter:      Mean: {current_row['flow_iat_mean']:<10.0f} us | Std Dev: {current_row['flow_iat_std']:<10.0f} us                         |")
    print(f"| Ground Truth at S(t): Attack Flow Share: {current_row['attack_ratio']:<6.2%} | Actual Class: {current_row['dominant_stage']:<22} |")
    print("+--------------------------------------------------------------------------------------+\n")

    # 6. Trajectory Forecast vs. Actual Ground Truth Comparison Table
    print("+---------------------------------------------------------------------------------------------------------+")
    print("| FORWARD TRAJECTORY SIMULATION: FORECAST vs. ACTUAL GROUND TRUTH                                         |")
    print("+---------+------------------+-----------------+--------+---------+-----------+----------------+--------------+")
    print(f"| {'Horizon':<7} | {'Forecast P(Atk)':<16} | {'Pred Stage':<15} | {'Conf':<6} | {'Actual?':<7} | {'Act Ratio':<9} | {'Actual Stage':<14} | {'Verification':<12} |")
    print("+---------+------------------+-----------------+--------+---------+-----------+----------------+--------------+")

    trajectory = results["trajectory"]

    for item in trajectory:
        step = item["step"]
        h_label = item["horizon_label"]
        pred_prob = item["attack_probability"]
        pred_stage = item["predicted_stage"]
        conf = item["stage_confidence"]

        # Ground truth for this step
        target_idx = eval_idx + step
        if target_idx < len(states_df):
            act_row = states_df.iloc[target_idx]
            act_attack = int(act_row.get("binary_attack", 0))
            act_ratio = act_row.get("attack_ratio", 0.0)
            act_stage = str(act_row.get("dominant_stage", "Normal"))
            act_status_str = "ATTACK" if act_attack == 1 else "BENIGN"
        else:
            act_attack = None
            act_ratio = None
            act_stage = "N/A"
            act_status_str = "N/A"

        # Verification Status
        is_pred_attack = (pred_prob >= args.threshold)
        if act_attack is not None:
            if is_pred_attack and act_attack == 1:
                verif = "TRUE POSITIVE"
            elif not is_pred_attack and act_attack == 0:
                verif = "TRUE NEGATIVE"
            elif is_pred_attack and act_attack == 0:
                verif = "EARLY WARN [!]"
            else:
                verif = "UNDERESTIMATE"
        else:
            verif = "PROJECTED"

        prob_str = f"{pred_prob:>6.1%}"
        conf_str = f"{conf:>5.1%}"
        ratio_str = f"{act_ratio:>7.2%}" if act_ratio is not None else "   N/A  "

        print(f"| {h_label:<7} | {prob_str:<16} | {pred_stage:<15} | {conf_str:<6} | {act_status_str:<7} | {ratio_str:<9} | {act_stage:<14} | {verif:<12} |")

    print("+---------+------------------+-----------------+--------+---------+-----------+----------------+--------------+\n")

    # 7. Forecast Warning Signal & Lead Time
    if results.get("has_warning"):
        print(f"[!] FORECAST WARNING: {results.get('warning_message', 'Attack progression detected.')}")
        if results.get("crossing_step"):
            lead_sec = int(results.get("crossing_step").replace("+", "")) * 5
            print(f"    Alert trigger at horizon {results.get('crossing_step')}. Estimated early warning lead time: {lead_sec}s.")
    else:
        print("[+] NOMINAL NETWORK TRAJECTORY: No attack progression predicted within horizon.")

    # 8. Top Contributing Features (Explainability)
    print("\n+--------------------------------------------------------------------------------------+")
    print("| EXPLAINABILITY: TOP ATTACK DRIVERS AT S(t)                                           |")
    print("+------------------------------+----------------+----------------+---------------------+")
    attributions = compute_feature_attributions(
        model=engine.model if engine.is_real_model else None,
        sequence_features=sequence_features,
        scaler=engine.scaler
    )

    print(f"| {'Feature Name':<28} | {'Observed':<14} | {'Baseline':<14} | {'Attribution (dP)':<19} |")
    print("+------------------------------+----------------+----------------+---------------------+")
    for attr in attributions[:5]:
        sign = "+" if attr["contribution"] > 0 else ""
        contrib_str = f"{sign}{attr['contribution']:<+6.4f}"
        effect = "[+] Increases Risk" if attr["contribution"] > 0 else "[-] Normalizing"
        name = attr['display_name'][:26]
        print(f"| {name:<28} | {attr['actual_value']:<14.4f} | {attr['baseline_value']:<14.4f} | {contrib_str:<8} {effect:<10} |")
    print("+------------------------------+----------------+----------------+---------------------+")

    print(f"\n[*] Run another window:  python forecast.py --scenario {args.scenario} --window <idx>")
    print(f"[*] List all windows:    python forecast.py --scenario {args.scenario} --list\n")

if __name__ == "__main__":
    run_cli()
