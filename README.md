# AI-Powered Network Attack Forecasting via World Models (SIH 2026)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph_Neural_Networks-3C2179)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-SOC_Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4_Compatible-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Predictive DFIR & Threat Trajectory Forecasting System**  
> An offline AI World Model that models continuous network graph dynamics over time:
>
> $$\mathcal{S}[t] \xrightarrow{\text{GINEConv Encoder}} z[t] \in \mathbb{R}^{128} \xrightarrow{\text{Latent GRU Rollout}} \hat{z}[t+1:t+K] \xrightarrow{\text{Threat Head}} \hat{y}[t+1:t+K] \xrightarrow{\text{ATT\&CK}} \text{Kill Chain \& SOAR}$$

---

## Table of Contents
- [1. Executive Summary & Paradigm Shift](#1-executive-summary--paradigm-shift)
- [2. System Architecture](#2-system-architecture)
- [3. Key Innovations & Capabilities](#3-key-innovations--capabilities)
- [4. Multi-Step Horizon Benchmark Results ($t+1 \dots t+5$)](#4-multi-step-horizon-benchmark-results)
- [5. Quick Start Guide](#5-quick-start-guide)
- [6. Live Red Team Attack Simulator](#6-live-red-team-attack-simulator)
- [7. Interactive SOC Decision-Support Dashboard](#7-interactive-soc-decision-support-dashboard)
- [8. Repository Structure](#8-repository-structure)
- [9. Hardware & VRAM Guardrails](#9-hardware--vram-guardrails)

---

## 1. Executive Summary & Paradigm Shift

Traditional Intrusion Detection Systems (NIDS) and Security Information and Event Management (SIEM) engines operate **reactively** on isolated packets, single log lines, or individual connection rows. This introduces fatal operational flaws:
1. **Zero Predictive Horizon**: Alerts trigger *after* exfiltration or lateral movement has already succeeded.
2. **Alert Fatigue (76%+ False Positive Rate)**: Static row-level classifiers flag benign packet bursts as attacks.
3. **Loss of Network Context**: Independent connection rows obscure host topologies, service dependencies, and multi-stage attack progressions.

### The World Model Solution
Our system treats the entire enterprise network as an **evolving continuous dynamical system**. By partitioning network flows into 30-second temporal graph windows $\mathcal{S}[t] = (\mathcal{V}, \mathcal{E})$, our GNN-GRU World Model learns the underlying transition dynamics in a compact 128-dimensional latent manifold. It autoregressively forecasts multi-step attack trajectories up to **150 seconds in advance**, isolating compromised hosts via GNNExplainer attribution and triggering automated SOAR containment playbooks before the adversary reaches their objective.

---

## 2. System Architecture

```
+---------------------------------------------------------------------------------------------------------+
|                                    INGESTION & GRAPH ENCODING PIPELINE                                  |
+---------------------------------------------------------------------------------------------------------+
  PCAP / CSV Traffic  ──>  Polars Streaming  ──>  30s Temporal Windows  ──>  Communication Subgraph S[t]
                                                                                │
                                              ┌─────────────────────────────────┘
                                              ▼
                           Node Features (9d): In/Out Degree, Byte Ratios, SYN/ACK, Entropy
                           Edge Features (7d): Duration, Packet Count, Volume, Rates, Flags
                                              │
                                              ▼
                             GINEConv 2-Layer Encoder + Dual Pooling
                                              │
                                              ▼
                                Latent State Vector: z[t] ∈ R^128
+---------------------------------------------------------------------------------------------------------+
|                                  AUTOREGRESSIVE LATENT WORLD MODEL                                      |
+---------------------------------------------------------------------------------------------------------+
       Historical Context: [z[t-9], z[t-8], ..., z[t]]
                                 │
                                 ▼
                     Latent Dynamics GRU Cell
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
  t+1 Latent State z^[t+1]                       Recursive Rollout
         │                                               │
         ▼                                               ▼
  Threat Prediction Head                      Horizon Forecasts: z^[t+2:t+5]
         │                                               │
         ▼                                               ▼
  Loss: Focal Loss (γ=1.5) + Cosine Latent Alignment + 2-Step Rollout Regularization
+---------------------------------------------------------------------------------------------------------+
|                               DECISION SUPPORT, EXPLAINABILITY & SOAR                                   |
+---------------------------------------------------------------------------------------------------------+
         │                                │                                │
         ▼                                ▼                                ▼
  GNNExplainer Subgraph          Conformal Prediction Sets         Automated SOAR Playbooks
  Edge Saliency & IP Attrib       Distribution-free 90-99%          iptables Rules & Suricata
```

---

## 3. Key Innovations & Capabilities

### 1. Dual-Objective Loss with 2-Step Unrolled Rollout
The World Model does not just fit the next step; it is trained using an autoregressive 2-step rollout loss that penalizes compounding simulation drift:
$$\mathcal{L} = \mathcal{L}_{\text{Focal}}(\hat{y}[t+1], y[t+1]; \gamma=1.5) + 0.5 \cdot (1 - \cos(\hat{z}[t+1], z[t+1])) + 0.5 \cdot \mathcal{L}_{\text{Focal}}(\hat{y}[t+2], y[t+2])$$

### 2. GNNExplainer Subgraph Saliency
Provides transparent edge attribution for any predicted threat. It computes the gradients of the predicted threat score with respect to the input graph adjacency and edge features, isolating the exact attacker IP, target host, and communication channel (e.g. Attacker `172.16.0.1` targeting Victim `192.168.10.50` on port 22 with 68.1% saliency).

### 3. Conformal Prediction Uncertainty Sets
Rather than emitting uncalibrated softmax probabilities, the system constructs distribution-free conformal prediction sets:
$$\mathcal{C}_{1-\alpha}(X) = \left\{ y \in \mathcal{Y} : s(X, y) \le \hat{q}_{1-\alpha} \right\}$$
Guaranteeing $90\%$, $95\%$, and $99\%$ statistical coverage bounds on out-of-distribution network shifts.

### 4. Automated SOAR Playbooks & Cyber Kill Chain Mapping
Predicted threat states are mapped in real time to the **MITRE ATT&CK** matrix and **Lockheed Martin Cyber Kill Chain** phases. The system dynamically generates:
- Priority containment workflows (immediate, secondary, post-incident).
- Ready-to-execute Linux `iptables` drop commands:
  ```bash
  iptables -I FORWARD 1 -s 172.16.0.1 -d 192.168.10.50 -p tcp --dport 22 -j DROP
  ```
- Copyable Suricata and Snort IDS signatures.

---

## 4. Multi-Step Horizon Benchmark Results

Evaluated on the out-of-time Test set across $t+1 \dots t+5$ prediction horizons (lead time of **30s to 150s**):

| Forecast Horizon | Lead Time | Baseline Macro F1 | **World Model Macro F1** | **Improvement** | Baseline FPR | **World Model FPR** | World Model Weighted F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$t+1$** | **30 sec** | 0.1784 | **0.3421** | **+91.8%** | 76.16% | **6.69%** | **81.19%** |
| **$t+2$** | **60 sec** | 0.1659 | **0.2829** | **+70.5%** | 76.45% | **11.05%** | **76.68%** |
| **$t+3$** | **90 sec** | 0.1471 | **0.2599** | **+76.7%** | 76.16% | **13.08%** | **75.07%** |
| **$t+4$** | **120 sec** | 0.1601 | **0.2405** | **+50.2%** | 75.87% | **13.95%** | **73.73%** |
| **$t+5$** | **150 sec** | 0.1748 | **0.2384** | **+36.4%** | 76.16% | **14.24%** | **74.19%** |

### Per-Class F1 Performance Highlights ($t+1$)
- **Benign Traffic**: **90.42%** F1 (vs Baseline 37.61%)
- **Brute Force (SSH/FTP)**: **86.96%** F1 (vs Baseline 18.82%)
- **Denial of Service (DoS/DDoS)**: **62.07%** F1 (vs Baseline 38.46%)
- **False Positive Rate**: Slashed from **76.16% down to 6.69%** (a 11.4× reduction in false alarms).

---

## 5. Quick Start Guide

### Prerequisites
- Python 3.11+
- Windows 10/11 or Linux
- CUDA-enabled GPU (NVIDIA RTX 30xx/40xx) or CPU

### Option A: One-Click Launch (Windows)
Double-click `start.bat` in the repository root. This automatically verifies the virtual environment, launches the Streamlit SOC Dashboard at `http://localhost:8501`, and opens your default browser.

```cmd
start.bat
```

### Option B: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Meghana-kb10/SIH_PROBLEM.git
   cd SIH_PROBLEM
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install torch_geometric polars pyyaml streamlit plotly networkx scikit-learn
   ```

4. **Launch the SOC Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

---

## 6. Live Red Team Attack Simulator

The repository includes a multi-scenario Red Team attack simulation engine that injects network traffic into socket listeners to test detection and forecasting capabilities in real time.

### Available Attack Scenarios
1. **`kill_chain`**: Full multi-stage attack (Reconnaissance $\to$ SSH Brute Force $\to$ Lateral Movement $\to$ Data Exfiltration).
2. **`brute_force`**: High-velocity Hydra-style SSH/FTP credential stuffing against port 22/21.
3. **`dos_flood`**: GoldenEye / HTTP Slowloris TCP SYN connection flood.
4. **`botnet_beacon`**: Periodic Mirai-style C2 heartbeats with jitter.

### Running via Web UI
In the Streamlit Dashboard sidebar:
1. Locate the **"Red Team Attack Simulator"** panel.
2. Select your attack scenario and target IP (`127.0.0.1` or remote target).
3. Click **"Launch Simulation"**. Progress, injected flows, and ATT&CK tactic IDs will stream directly in the interface.

### Running via CLI
```bash
python scripts/simulate_live_attack.py --scenario kill_chain --target 127.0.0.1 --intensity 3
```

---

## 7. Interactive SOC Decision-Support Dashboard

The dark-themed Streamlit SOC interface (`dashboard/app.py`) provides:

1. **Operational Overview**: Real-time current threat state, threat probability gauge, and alert tier (Critical, High, Medium, Low).
2. **Interactive Topology Visualizer**: NetworkX + Plotly interactive graph displaying communication subgraphs, node degrees, and anomalous link highlighting.
3. **Multi-Step Horizon Forecaster**: Side-by-side comparison of actual vs forecasted trajectory progression across steps $t+1 \dots t+5$.
4. **Timeline Replay Engine**: Step through historical network windows with Play, Pause, and Step controls.
5. **GNN Subgraph Attribution**: Interactive bar chart and saliency metrics ranking malicious edge interactions.
6. **Conformal Prediction Sets**: Visual coverage intervals with configurable confidence bounds ($90\%, 95\%, 99\%$).
7. **Automated SOAR Playbooks**: Copyable firewall commands and incident response containment checklists.
8. **One-Click Incident Report**: Exports an executive DFIR incident response summary in Markdown format.

---

## 8. Repository Structure

```
SIH_PROBLEM/
├── configs/
│   └── config.yaml                     # Centralized paths, hyperparameters, and feature dimensions
├── data/
│   ├── cic2017/                        # Raw dataset CSVs (excluded via .gitignore)
│   ├── processed/                      # Temporal window partitions (train/val/test)
│   └── graphs/                         # Serialized PyG graph datasets (train/val/test)
├── models/
│   ├── world_model.pt                  # Pretrained GNN+GRU World Model checkpoint (2.96 MB)
│   └── baseline_model.pkl              # Baseline logistic regression benchmark
├── src/
│   ├── adapters/
│   │   ├── base_adapter.py             # Strongly-typed FlowRecord dataclass
│   │   ├── cic_ids2017.py              # High-speed Polars CSV streaming adapter
│   │   └── pcap_adapter.py             # PCAP binary reader interface
│   ├── preprocessing/
│   │   ├── cleaning.py                 # Outlier hygiene and median imputation
│   │   ├── windowing.py                # Contiguous 30-second temporal window generator
│   │   └── normalization.py            # Train-only fitted RobustScalers
│   ├── graph/
│   │   ├── features.py                 # 9d node & 7d edge feature extractors
│   │   ├── sampling.py                 # MAX_EDGES=10,000 priority guardrail
│   │   └── builder.py                  # PyTorch Geometric Data graph builder
│   ├── models/
│   │   ├── gnn_encoder.py              # 2-layer GINEConv spatial encoder
│   │   ├── world_model.py              # Latent GRU World Model & Focal Loss
│   │   ├── prediction_head.py          # Threat classification head
│   │   └── baseline.py                 # Logistic regression baseline
│   ├── forecasting/
│   │   ├── rollout.py                  # Recursive K-step autoregressive rollout engine
│   │   ├── risk.py                     # Trajectory risk aggregator
│   │   ├── mitre_mapping.py            # Transparent MITRE ATT&CK mapper
│   │   ├── kill_chain.py               # Cyber Kill Chain progression analyzer
│   │   ├── conformal.py                # Distribution-free conformal prediction sets
│   │   └── soar_playbooks.py           # Automated SOAR containment playbook generator
│   └── explainability/
│       ├── attribution.py              # Gradient feature attribution
│       └── subgraph_explainer.py       # GNNExplainer edge & host attribution
├── scripts/
│   ├── inspect_dataset.py              # Dataset validation and topology analysis
│   ├── preprocess.py                   # Chronological block dataset partitioning
│   ├── build_graphs.py                 # PyG graph serialization
│   ├── train_world_model.py            # 2-step rollout GPU training script
│   ├── train_baseline.py               # Baseline benchmark trainer
│   ├── evaluate.py                     # Multi-step horizon benchmark evaluation
│   ├── simulate_live_attack.py         # Standalone Red Team socket attack simulator
│   └── run_forecast.py                 # CLI demonstration script
├── dashboard/
│   └── app.py                          # Streamlit SOC Command Center
├── reports/
│   ├── dataset_inspection.json         # Raw dataset audit & topology statistics
│   └── evaluation_report.json          # Benchmark evaluation report (t+1 ... t+5)
├── .streamlit/
│   └── config.toml                     # Streamlit server & theme configuration
├── start.bat                           # One-click Windows runner
└── README.md                           # Documentation
```

---

## 9. Hardware & VRAM Guardrails

- **Dynamic Edge Sampling**: Enforces a strict `MAX_EDGES = 10,000` limit using priority sampling (prioritizing high-volume, high-SYN, and diverse ports). This eliminates out-of-memory errors on consumer GPUs with 4 GB to 8 GB VRAM.
- **Fixed Latent Dimensionality**: Graph topology of arbitrary size ($|V|, |E|$) is continuously condensed into a compact $128$-dimensional vector $z[t]$ (concatenation of 64d global mean and 64d global max pooling).
- **Mixed Precision Acceleration**: Training and evaluation leverage `torch.amp` (FP16/BF16) for high throughput on NVIDIA Tensor Cores.
- **Graceful CPU Fallback**: Automatically switches to CPU if an NVIDIA GPU is not detected.

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
