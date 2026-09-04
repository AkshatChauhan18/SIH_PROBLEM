# 🛡️ AI World Model Network Attack Forecasting — Model Training Guide
**SIH 2026 — Problem ID: SIH26153**  
**Target Architecture**: Offline Temporal Graph World Model ($S[t] \to z[t] \in \mathbb{R}^{128} \to \hat{z}[t+1:t+K]$)  
**Target Hardware**: NVIDIA GeForce RTX 4060 (8 GB VRAM), 24 GB RAM, Windows / PowerShell

---

## Table of Contents
1. [Overview & Prerequisites](#1-overview--prerequisites)
2. [Full Training Pipeline Workflow](#2-full-training-pipeline-workflow)
3. [Step 1 — Preprocess Raw Flows into Temporal Windows](#step-1--preprocess-raw-flows-into-temporal-windows)
4. [Step 2 — Construct & Scale PyG Graph Windows](#step-2--construct--scale-pyg-graph-windows)
5. [Step 3 — Train Flat Statistical Baseline](#step-3--train-flat-statistical-baseline)
6. [Step 4 — Train GNN Encoder + GRU World Model](#step-4--train-gnn-encoder--gru-world-model)
7. [Step 5 — Run Multi-Step Horizon Benchmark Evaluation](#step-5--run-multi-step-horizon-benchmark-evaluation)
8. [Step 6 — Test Forecast Rollout Demonstration (CLI)](#step-6--test-forecast-rollout-demonstration-cli)
9. [Step 7 — Launch Offline SOC Decision Support Dashboard](#step-7--launch-offline-soc-decision-support-dashboard)
10. [Hardware & Hyperparameter Tuning](#10-hardware--hyperparameter-tuning)
11. [Troubleshooting & FAQs](#11-troubleshooting--faqs)

---

## 1. Overview & Prerequisites

### Core Differentiation
Unlike traditional intrusion detection systems that perform static row-by-row classification, this system models **network state dynamics**:
- Represents each 30-second traffic window as a directed multigraph $S[t] = (V_t, E_t)$.
- Encodes topology into an invariant latent state $z[t] \in \mathbb{R}^{128}$ via a 2-layer `GINEConv` encoder.
- Learns latent transition dynamics $P(z[t+1] \mid z[t-N+1:t])$ via a 2-layer GRU.
- Forecasts future attack progression recursively up to $K=5$ steps (150 seconds advance warning).
- Maps predictions to MITRE ATT&CK tactics, techniques, and SOC mitigation actions.

### Zero Data Leakage Guarantee
- Imputation medians and feature scalers (`RobustScaler`) are **strictly fitted on the Train partition (Days 1–3) only**.
- Chronological partitioning:
  - **Train (Days 1–3)**: `Monday`, `Tuesday`, `Wednesday`
  - **Validation (Day 4)**: `Thursday-Morning-WebAttacks`, `Thursday-Afternoon-Infilteration`
  - **Test (Day 5)**: `Friday-Morning`, `Friday-Afternoon-PortScan`, `Friday-Afternoon-DDos`
- Target labels are **never** supplied as graph node or edge features.

### Python Environment
All commands assume the project virtual environment at `d:\sih_project\.venv`. Run all commands from the repository root `d:\sih_project`.

---

## 2. Full Training Pipeline Workflow

```
[Raw CSVs in data/cic2017/]
             │
             ▼
      (Step 1: preprocess.py)
             │
             ├── models/preprocessors.pkl (Train medians)
             └── data/processed/{train,val,test}_windows.pkl
             │
             ▼
     (Step 2: build_graphs.py)
             │
             ├── models/preprocessors.pkl (Updated with RobustScalers)
             └── data/graphs/{train,val,test}_graphs.pt (PyG S[t], MAX_EDGES=10k)
             │
      ┌──────┴────────────────────────┐
      ▼                               ▼
(Step 3: train_baseline.py)     (Step 4: train_world_model.py)
      │                               │
models/baseline_model.pkl       models/world_model.pt (Dual Loss)
      │                               │
      └──────────────┬────────────────┘
                     ▼
           (Step 5: evaluate.py)
                     │
           reports/evaluation_report.json (t+1..t+5 Metrics)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
(Step 6: run_forecast.py)   (Step 7: streamlit run dashboard/app.py)
```

---

## Step 1 — Preprocess All 8 CSVs into Chronological Temporal Windows

Partitions all 8 CIC-IDS2017 raw flow datasets into continuous 30-second temporal windows with a strict **chronological 70% Train / 15% Val / 15% Test split per file**. This ensures the model learns **all 7 attack families** (DDoS, Botnet, PortScan, Infiltration, Web Attacks, Brute Force, Benign) while guaranteeing zero data leakage (the past predicts the future).

### Command
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/preprocess.py --config configs/config.yaml
```

### What Happens:
1. Loads all 8 raw CSV files with Polars lazy streaming (`utf8-lossy`).
2. Cleans non-ASCII column headers and adjusts 12h working hours timestamps to continuous 24h format.
3. Fits feature medians on the first 70% of each day's flows (Zero Leakage) and saves them to `models/preprocessors.pkl`.
4. Windows all flows into 30-second slices ($W_t$) and partitions each file:
   - **First 70% of time slices** $\to$ `train_windows.pkl`
   - **Next 15% of time slices** $\to$ `val_windows.pkl`
   - **Final 15% of time slices** $\to$ `test_windows.pkl`

### Expected Outputs:
- `models/preprocessors.pkl`
- `data/processed/train_windows.pkl` (~2,100 windows across all 7 attacks)
- `data/processed/val_windows.pkl` (~450 windows)
- `data/processed/test_windows.pkl` (~450 windows)

---

## Step 2 — Construct & Scale PyG Graph Windows

Converts window records into PyTorch Geometric `Data` graphs $S[t]$. Applies the deterministic priority edge sampling guardrail (`MAX_EDGES=10,000`) to guarantee RTX 4060 GPU VRAM safety, and fits `RobustScaler` on Train graphs only.

### Command
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/build_graphs.py --config configs/config.yaml
```

### What Happens:
1. Extracts 9 statistical node features (degrees, packet/byte rates, SYN/RST ratios, peer count) and 7 edge features (flows, bytes, packets, duration, IAT, SYN count, port entropy).
2. If a window exceeds 10,000 edges, deterministically retains critical attack signals (high volume, high SYN, high port entropy) without random edge drop.
3. Fits `RobustScaler` on Train graph node and edge features and saves updated scalers to `models/preprocessors.pkl`.
4. Serializes scaled PyG graph lists to disk.

### Expected Outputs:
- `data/graphs/train_graphs.pt`
- `data/graphs/val_graphs.pt`
- `data/graphs/test_graphs.pt`

---

## Step 3 — Train Flat Statistical Baseline

Trains a standard Logistic Regression benchmark on flat temporal-window aggregated statistics. This serves as the comparison baseline required to evaluate the World Model's advance forecast lead time and graph-modeling superiority.

### Command
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/train_baseline.py --config configs/config.yaml
```

### Expected Outputs:
- `models/baseline_model.pkl`
- Console evaluation report on Validation data.

---

## Step 4 — Train GNN Encoder + GRU World Model

Trains the end-to-end AI World Model on your **NVIDIA GeForce RTX 4060 GPU** using PyTorch mixed precision (`torch.amp`), vectorized PyG batching, sequence shuffling, and anti-collapse dual loss.

The network optimizes the **Anti-Collapse Dual Loss**:
$$\mathcal{L} = \text{MSE}(\hat{z}[t+1], \text{sg}(z[t+1])) + \lambda \times \left[0.5 \cdot \text{CE}(z[t+1], y) + 0.5 \cdot \text{CE}(\hat{z}[t+1], y)\right]$$

### Command
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/train_world_model.py --config configs/config.yaml
```

*(This automatically uses `batch_size: 32`, `epochs: 40`, `lambda_threat: 1.0`, sequence shuffling, and vectorized PyG batching for ~7s/epoch speed).*

### Optional Command Flags:
| Flag | Description | Default | Recommended for Fast Test |
|------|-------------|---------|---------------------------|
| `--epochs` | Number of training epochs | `40` | `5` for quick test |
| `--batch-size` | Sequence batch size | `32` | `32` |
| `--config` | Custom configuration file | `configs/config.yaml` | — |

*Example of a quick 5-epoch test run:*
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/train_world_model.py --config configs/config.yaml --epochs 5 --batch-size 64
```

### Expected Outputs:
- Best validation model checkpoint saved to `models/world_model.pt`.
- Per-epoch metrics displayed: MSE Loss, Classification Loss, Total Dual Loss, Validation Macro F1.

---

## Step 5 — Run Multi-Step Horizon Benchmark Evaluation

Benchmarks the trained World Model against the Baseline on the held-out **Test set (Day 5 - Friday)** across forecast horizons $t+1, t+2, t+3, t+4, t+5$ (30s to 150s ahead).

### Command
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/evaluate.py --config configs/config.yaml --output reports/evaluation_report.json
```

### Metrics Evaluated:
- **Macro F1 Score** per horizon step.
- **Per-Class F1, Precision, and Recall** (detecting stealthy infiltration vs volumetric DDoS).
- **False Positive Rate (FPR)** on benign traffic.
- **Advance Forecast Lead Time Gain** over the reactive baseline.

### Expected Outputs:
- `reports/evaluation_report.json`
- Side-by-side terminal comparison table ($t+1 \dots t+5$).

---

## Step 6 — Test Forecast Rollout Demonstration (CLI)

Performs a live CLI demonstration selecting an active test window $S[t]$, rolling out recursive latent predictions $\hat{z}[t+1:t+5]$, and verifying them against actual ground truth.

### Command
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/run_forecast.py --config configs/config.yaml --window-idx 25 --horizon 5
```

### What You Will See:
- Active network state statistics at window $t$.
- Step-by-step forecasted attack behavior vs actual future ground truth.
- Threat score trajectory ($P(\text{Threat})$) and SOC Risk Tiers (`NOMINAL`, `ELEVATED`, `CRITICAL`).
- MITRE ATT&CK tactic/technique mapping and recommended containment action.
- Live gradient feature and temporal influence attributions.

---

## Step 7 — Launch Offline SOC Decision Support Dashboard

Launches the offline Cybersecurity Operations Center command center in your browser.

### Command
```powershell
& "d:\sih_project\.venv\Scripts\streamlit.exe" run dashboard/app.py
```

### Dashboard Features:
- **URL**: `http://localhost:8501`
- **Telemetry Bar**: Active hosts $|V|$, directed connections $|E|$, current state label, and real-time threat probability.
- **Interactive Trajectory**: Autoregressive rollout curve comparing forecasted threat evolution against actual ground truth.
- **Subnet Graph Topology**: Interactive Plotly/NetworkX visualization of the communication graph.
- **MITRE ATT&CK Matrix**: Tactical profiling and actionable incident response recommendations.
- **Gradient Attribution**: Bar chart of top feature drivers and line plot of temporal timeline influence ($t-N \dots t$).

---

## 10. Hardware & Hyperparameter Tuning

All parameters are configured in [configs/config.yaml](file:///d:/sih_project/configs/config.yaml):

```yaml
temporal_windowing:
  window_seconds: 30     # 30-second temporal slices
  sequence_length: 10    # N = 10 past windows (300s context)
  forecast_horizon: 5    # K = 5 recursive rollout steps (150s lead time)

graph_construction:
  max_edges: 10000       # Priority sampling guardrail for 8GB VRAM
  node_feature_dim: 9
  edge_feature_dim: 7
  flat_feature_dim: 8

model_architecture:
  gnn_hidden_dim: 64
  latent_dim: 128        # Invariant latent dimension: 64 mean pool + 64 max pool
  gru_hidden_dim: 128
  gru_num_layers: 2
  dropout: 0.1

training:
  batch_size: 32         # Fits comfortably in 8GB RTX 4060 VRAM
  learning_rate: 0.001
  epochs: 20
  lambda_threat: 1.0     # Loss = MSE + lambda * CE
  use_amp: true          # PyTorch AMP mixed precision
  device: "cuda"         # Auto-detects NVIDIA RTX 4060
```

---

## 11. Troubleshooting & FAQs

### Q1: `CUDA out of memory` during training
**Fix**: In [configs/config.yaml](file:///d:/sih_project/configs/config.yaml), reduce `batch_size` from `32` to `16` or run with `--batch-size 16`. The `MAX_EDGES=10,000` guardrail prevents individual graph explosion, so reducing batch size directly frees VRAM.

### Q2: How long does Step 1 take?
Polars parallel streaming processes the ~3.1 million raw flows in approximately **1–2 minutes** depending on disk read speeds.

### Q3: Can I run on CPU if needed?
Yes. The device detection in `src/utils/device.py` automatically falls back to CPU if CUDA is unavailable. To force CPU, set `device: "cpu"` in `configs/config.yaml`.

### Q4: Are the target labels ever leaked to the GNN?
No. Node features (9d) and edge features (7d) are calculated purely from flow network statistics. The target label is assigned to `Data.y` only and is evaluated strictly at the ThreatHead output after latent state forecasting.
