"""
SENTINEL-X Configuration and Domain Logic
SIH26153 - AI based Network Attack Forecasting from Network Traffic Data
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "cic2017"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "world_model.pth"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# MITRE ATT&CK Phase / Attack Stage Mappings
STAGE_NAMES = [
    "Normal",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Impact"
]

STAGE_COLORS = {
    "Normal": "#00E676",            # Bright Green
    "Reconnaissance": "#00B0FF",     # Cyan / Sky Blue
    "Initial Access": "#FFD600",     # Amber / Gold
    "Lateral Movement": "#FF9100",    # Orange
    "Command & Control": "#FF3D00",   # Deep Orange / Red
    "Impact": "#D50000"              # Crimson Red
}

LABEL_TO_STAGE = {
    "BENIGN": "Normal",
    "PortScan": "Reconnaissance",
    "FTP-Patator": "Initial Access",
    "SSH-Patator": "Initial Access",
    "Brute Force": "Initial Access",
    "Web Attack – Brute Force": "Initial Access",
    "Web Attack - Brute Force": "Initial Access",
    "Web Attack – XSS": "Initial Access",
    "Web Attack - XSS": "Initial Access",
    "Web Attack – SQL Injection": "Initial Access",
    "Web Attack - SQL Injection": "Initial Access",
    "Heartbleed": "Initial Access",
    "Infiltration": "Lateral Movement",
    "Bot": "Command & Control",
    "DDoS": "Impact",
    "DoS GoldenEye": "Impact",
    "DoS Hulk": "Impact",
    "DoS Slowhttptest": "Impact",
    "DoS slowloris": "Impact",
}

STAGE_TO_ID = {name: idx for idx, name in enumerate(STAGE_NAMES)}
ID_TO_STAGE = {idx: name for idx, name in enumerate(STAGE_NAMES)}

# State feature definitions (extracted per 5-second temporal aggregation window)
STATE_FEATURES = [
    "total_flows",
    "total_packets",
    "total_bytes",
    "unique_source_ips",
    "unique_dest_ips",
    "unique_dest_ports",
    "port_diversity",
    "dest_diversity",
    "syn_ratio",
    "ack_ratio",
    "rst_ratio",
    "fin_ratio",
    "psh_ratio",
    "flow_packets_per_sec",
    "flow_bytes_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "average_packet_size",
    "down_up_ratio",
    "active_mean",
    "idle_mean",
    "init_window_forward_mean",
    "init_window_backward_mean",
]

# Friendly display names for UI
FEATURE_DISPLAY_NAMES = {
    "total_flows": "Total Flows",
    "total_packets": "Total Packets",
    "total_bytes": "Total Volume (Bytes)",
    "unique_source_ips": "Source IP Count",
    "unique_dest_ips": "Dest IP Count",
    "unique_dest_ports": "Unique Dest Ports",
    "port_diversity": "Port Diversity",
    "dest_diversity": "Destination Diversity",
    "syn_ratio": "SYN Packet Ratio",
    "ack_ratio": "ACK Packet Ratio",
    "rst_ratio": "RST Packet Ratio",
    "fin_ratio": "FIN Packet Ratio",
    "psh_ratio": "PSH Packet Ratio",
    "flow_packets_per_sec": "Packets / sec",
    "flow_bytes_per_sec": "Bytes / sec",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std Dev",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "packet_length_mean": "Packet Length Mean",
    "packet_length_std": "Packet Length Std Dev",
    "packet_length_variance": "Packet Length Variance",
    "average_packet_size": "Average Packet Size",
    "down_up_ratio": "Down/Up Ratio",
    "active_mean": "Active Time Mean",
    "idle_mean": "Idle Time Mean",
    "init_window_forward_mean": "Init TCP Win Fwd",
    "init_window_backward_mean": "Init TCP Win Bwd",
}

# Default Hyperparameters
DEFAULT_CONFIG = {
    "window_size_seconds": 5,
    "sequence_length": 10,
    "forecast_horizon": 5,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "attack_threshold": 0.50,
}

# CSV Dataset registry
DATASET_FILES = {
    "Friday PortScan (Reconnaissance)": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday DDoS (Impact)": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday Morning (BENIGN / Normal)": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday Web Attacks (Initial Access)": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday Infiltration (Lateral Movement)": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Tuesday Brute Force / Patator": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday DoS / Heartbleed": "Wednesday-workingHours.pcap_ISCX.csv",
    "Monday Baseline (BENIGN)": "Monday-WorkingHours.pcap_ISCX.csv",
}
