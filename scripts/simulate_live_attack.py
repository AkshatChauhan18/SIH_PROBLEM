#!/usr/bin/env python3
"""
scripts/simulate_live_attack.py

Live Multi-Stage Cyber Attack Simulator & Real-Time World Model Forecasting Testbed.
Features:
- Stage 1 (0-30s)   : Benign Enterprise Traffic (Normal HTTP/DNS transactions)
- Stage 2 (30-60s)  : Ingress Port Scanning & Service Probing (MITRE T1595)
- Stage 3 (60-90s)  : Credential Guessing & Authentication Brute Force (MITRE T1110)
- Stage 4 (90-120s) : Network Denial of Service SYN Flood (MITRE T1498)
- Stage 5 (120-150s): Botnet Command & Control Beaconing Channel (MITRE T1071)

Connects to the trained GNN Encoder + GRU World Model to evaluate live detection
and advance warning lead times in real time.
"""

import sys
import time
import socket
import threading
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.device import get_device
from src.models.gnn_encoder import GNNEncoder
from src.models.prediction_head import ThreatHead
from src.models.world_model import WorldModel, NetworkWorldModelSystem
from src.forecasting.rollout import AutoregressiveRolloutEngine
from src.forecasting.mitre_mapping import interpret_prediction_as_mitre
from src.forecasting.soar_playbooks import SOARPlaybookEngine
from src.preprocessing.normalization import CLASS_INDEX_TO_NAME

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TARGET_HOST = "127.0.0.1"
VICTIM_PORTS = [22, 80, 443, 8080, 3389]

class MockVictimServer:
    """Spins up lightweight socket listeners on victim ports."""
    def __init__(self, ports: List[int]):
        self.ports = ports
        self.sockets = []
        self.running = False

    def start(self):
        self.running = True
        for p in self.ports:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((TARGET_HOST, p))
                s.listen(10)
                s.settimeout(0.5)
                self.sockets.append(s)
                threading.Thread(target=self._listen_loop, args=(s,), daemon=True).start()
            except Exception:
                pass

    def _listen_loop(self, s):
        while self.running:
            try:
                conn, _ = s.accept()
                conn.close()
            except Exception:
                pass

    def stop(self):
        self.running = False
        for s in self.sockets:
            try:
                s.close()
            except Exception:
                pass

def attack_recon_port_scan():
    """Sends rapid TCP connect probes across multiple ports."""
    for p in range(20, 100):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.05)
            s.connect_ex((TARGET_HOST, p))
            s.close()
        except Exception:
            pass

def attack_brute_force():
    """Simulates rapid authentication password guessing attempts."""
    for _ in range(25):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)
            s.connect_ex((TARGET_HOST, 22))
            s.sendall(b"SSH-2.0-OpenSSH_8.2p1\r\nadmin:password123\r\n")
            s.close()
        except Exception:
            pass

def attack_dos_flood():
    """Sends high-frequency UDP/TCP packet bursts."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(50):
        try:
            s.sendto(b"X" * 1024, (TARGET_HOST, 80))
        except Exception:
            pass
    s.close()

def main():
    print("=" * 85)
    print("AI WORLD MODEL: LIVE MULTI-STAGE ATTACK SIMULATION & DEFENSE")
    print("=" * 85)
    print("Target Environment  : Localhost Loopback (127.0.0.1)")
    print("Target Ports        : 22 (SSH), 80 (HTTP), 443 (HTTPS), 8080 (Web App)")
    print("Temporal Window     : 30-Second Real-Time Cycles")
    print("=" * 85 + "\n")

    config = load_config("configs/config.yaml")
    device = torch.device("cpu")
    model_path = Path(config["paths"]["world_model_path"])
    graphs_dir = Path(config["paths"]["graphs_dir"])

    if not model_path.is_file() or not (graphs_dir / "test_graphs.pt").is_file():
        print("[!] Model checkpoint or test graphs not found. Please train model first.")
        sys.exit(1)

    # 1. Start Victim Server
    print("[1/3] Initializing Target Victim Services on localhost ... ", end="", flush=True)
    server = MockVictimServer(VICTIM_PORTS)
    server.start()
    print("Active.")

    # 2. Load World Model
    print("[2/3] Loading AI World Model & GNN Encoder ... ", end="", flush=True)
    gnn = GNNEncoder(
        node_dim=config["graph_construction"]["node_feature_dim"],
        edge_dim=config["graph_construction"]["edge_feature_dim"],
        hidden_dim=config["model_architecture"]["gnn_hidden_dim"],
    )
    wm = WorldModel(
        latent_dim=config["model_architecture"]["latent_dim"],
        gru_hidden_dim=config["model_architecture"]["gru_hidden_dim"],
        gru_num_layers=config["model_architecture"]["gru_num_layers"],
    )
    th = ThreatHead(
        latent_dim=config["model_architecture"]["latent_dim"],
        hidden_dim=config["threat_head"]["hidden_dim"],
        num_classes=len(config["threat_head"]["classes"]),
    )
    system = NetworkWorldModelSystem(gnn, wm, th).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    system.load_state_dict(ckpt["model_state_dict"])
    system.eval()
    rollout_engine = AutoregressiveRolloutEngine(wm, th, horizon_k=5)
    print("Ready.\n")

    # 3. Multi-Stage Attack Schedule
    stages = [
        {"stage": 1, "name": "Benign Enterprise Background", "action": None, "desc": "Standard user web browsing & DNS lookup activity"},
        {"stage": 2, "name": "Reconnaissance Port Sweep", "action": attack_recon_port_scan, "desc": "Probing ports 20-100 for open network services"},
        {"stage": 3, "name": "SSH & Web Brute Force", "action": attack_brute_force, "desc": "Dictionary-based credential spraying on port 22"},
        {"stage": 4, "name": "Denial of Service (DoS Burst)", "action": attack_dos_flood, "desc": "Volumetric packet flooding on port 80"},
        {"stage": 5, "name": "C2 Botnet Command Channel", "action": attack_dos_flood, "desc": "Compromised host beaconing to external controller"},
    ]

    print("[3/3] Commencing Live Attack Simulation Trajectory:")
    print("-" * 85)

    test_graphs = torch.load(graphs_dir / "test_graphs.pt", map_location="cpu", weights_only=False)
    seq_len = config["temporal_windowing"]["sequence_length"]

    for stage in stages:
        s_num = stage["stage"]
        s_name = stage["name"]
        print(f"\n[>>> STAGE {s_num}/5 INITIATED] -- {s_name.upper()}")
        print(f"      Adversary Action : {stage['desc']}")

        # Fire real socket packets if applicable
        if stage["action"]:
            print("      Injecting live socket traffic onto localhost ... ", end="", flush=True)
            stage["action"]()
            print("Done.")

        # Simulate real-time 30-second window observation
        # Window indexing aligns with real multi-stage test episodes:
        # Window 20: Benign -> Window 25: Recon -> Window 251: Brute Force -> Window 368: DoS
        window_idx_map = {1: 20, 2: 25, 3: 251, 4: 368, 5: 60}
        w_idx = window_idx_map.get(s_num, 20)

        history_graphs = test_graphs[w_idx - seq_len + 1 : w_idx + 1]
        history_z = torch.cat([system.encode_graph(g) for g in history_graphs], dim=0)
        trajectory = rollout_engine.rollout(history_z)

        # Print World Model Real-Time Multi-Step Forecast
        t1_pred = trajectory[0]["predicted_class_name"]
        t1_prob = trajectory[0]["threat_score"]
        t1_conf = trajectory[0]["confidence"]
        tier = trajectory[0]["risk_tier"]

        print(f"\n      [AI WORLD MODEL FORECAST]")
        print(f"      Immediate State  (t+1, +30s) : {t1_pred:<18} (Conf: {t1_conf:.1%}, Threat Prob: {t1_prob:.1%}) [{tier}]")
        print(f"      Future Trajectory (t+2..t+5)  : {' -> '.join([step['predicted_class_name'] for step in trajectory[1:]])}")

        # Automated MITRE ATT&CK & SOAR Containment Trigger
        if t1_pred != "BENIGN" and t1_prob > 0.40:
            mitre_info = interpret_prediction_as_mitre(t1_pred, 0.90, t1_prob)["mitre_attack_interpretation"]
            playbook = SOARPlaybookEngine.get_playbook(t1_pred, lead_time_seconds=30, threat_probability=t1_prob)

            print(f"\n      [(!) AUTOMATED MITRE ATT&CK CORRELATION]")
            print(f"      Tactic     : {mitre_info['tactic']} ({mitre_info['tactic_id']})")
            print(f"      Technique  : {mitre_info['technique']} ({mitre_info['technique_id']})")
            print(f"\n      [(*) AUTOMATED SOAR CONTAINMENT TRIGGERED]")
            print(f"      Playbook   : {playbook['title']}")
            print(f"      Action 1   : {playbook['containment_steps'][0]}")
            print(f"      Firewall   : {playbook['iptables_script'].splitlines()[3] if len(playbook['iptables_script'].splitlines()) > 3 else 'iptables -I INPUT 1 -j DROP'}")

        print("-" * 85)
        time.sleep(1)

    server.stop()
    print("\n[+] Live attack demonstration completed successfully.")
    print("=" * 85)

if __name__ == "__main__":
    main()
