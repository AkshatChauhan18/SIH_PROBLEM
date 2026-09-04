# AI-Based Network Attack Forecasting via World Model (SIH26153)

> **SIH 2026 — Cybersecurity & AI / World Models**  
> An offline AI World Model that models evolving network state dynamics:
> $$S[t] \xrightarrow{\text{Latent Dynamics}} P(z[t+1] \mid z[t-N+1:t]) \xrightarrow{\text{Forecast}} z[t+1:t+K] \xrightarrow{\text{Interpretation}} \text{Attack Progression}$$

---

## 1. Core Architecture & Differentiation

This system is **NOT a static row-level classifier**. Rather than classifying individual packets or isolated connection rows, the system models continuous network graph states $S[t]$ over 30-second temporal windows:

1. **Adapter Layer**: Normalizes raw network traffic (CSVs or PCAP) into strongly-typed `FlowRecord` objects.
2. **Temporal Graph Construction**: Aggregates flows into directed communication subgraphs $S[t] = (V, E)$ with 9-dimensional statistical node features and 7-dimensional edge features.
3. **Graph VRAM Guardrail**: Deterministic priority edge sampler enforcing `MAX_EDGES = 10,000` to preserve GPU memory on NVIDIA RTX 4060 (8 GB VRAM).
4. **2-Layer GINEConv Encoder**: Projects graph topology into a fixed **128-dimensional** latent state vector $z[t]$ (concatenating 64d Global Mean Pooling and 64d Global Max Pooling).
5. **GRU World Model**: Models temporal dynamics in latent space, learning $P(z[t+1] \mid z[t-N+1:t])$ over the past $N=10$ windows.
6. **Autoregressive K-Step Rollout**: Genuine recursive forecasting predicting future latent states $\hat{z}[t+1], \dots, \hat{z}[t+K]$ ($K=5$ steps / 150s lead time).
7. **Threat Head & Dual Loss**: Couples latent dynamics loss with classification loss:
   $$\mathcal{L} = \text{MSE}(\hat{z}[t+1], z[t+1]) + \lambda \times \text{CrossEntropy}(\text{ThreatHead}(\hat{z}[t+1]), y[t+1])$$
8. **MITRE ATT&CK Mapping**: A separate, transparent interpretation layer mapping predicted operational states to MITRE tactics and techniques.
9. **Gradient Explainability**: Computes real gradients of predicted future threat scores back to input feature drivers and temporal states ($t-N \dots t$).
10. **Streamlit SOC Dashboard**: Offline dark-themed Cybersecurity Operations Center decision-support interface.

---

## 2. Directory Layout

```
d:/sih_project/
├── configs/
│   └── config.yaml                     # Centralized hyperparameter and paths configuration
├── data/
│   ├── cic2017/                        # Raw CSV files (preserved untouched)
│   ├── processed/                      # Preprocessed temporal window partitions
│   └── graphs/                         # Serialized PyG Data graph collections
├── src/
│   ├── adapters/
│   │   ├── base_adapter.py             # FlowRecord dataclass & BaseAdapter interface
│   │   ├── cic_ids2017.py              # Polars streaming adapter with 24h timestamp parser
│   │   └── pcap_adapter.py             # Extensible PCAP ingestion interface
│   ├── preprocessing/
│   │   ├── cleaning.py                 # Data hygiene & median imputation
│   │   ├── windowing.py                # Chronological temporal window partitioner
│   │   └── normalization.py            # Train-only fitted scalers & label mapping
│   ├── graph/
│   │   ├── features.py                 # Node (9d) and edge (7d) feature extraction
│   │   ├── sampling.py                 # Deterministic MAX_EDGES=10,000 priority guardrail
│   │   └── builder.py                  # PyG Data graph builder S[t]
│   ├── models/
│   │   ├── gnn_encoder.py              # 2-layer GINEConv encoder (z in R^128)
│   │   ├── world_model.py              # GRU World Model & Dual Loss system
│   │   ├── prediction_head.py          # ThreatHead multi-class event classifier
│   │   └── baseline.py                 # Baseline Logistic Regression on flat window stats
│   ├── forecasting/
│   │   ├── rollout.py                  # Recursive K-step autoregressive rollout engine
│   │   ├── risk.py                     # Horizon risk & infiltration probability
│   │   └── mitre_mapping.py            # Rule-based MITRE ATT&CK interpretation layer
│   ├── explainability/
│   │   └── attribution.py              # Gradient-based feature & temporal attribution
│   └── utils/
│       ├── config.py                   # YAML configuration loader
│       ├── device.py                   # Device selection & torch.amp mixed precision helper
│       └── metrics.py                  # Macro F1, Per-class F1, FPR, Lead Time
├── scripts/
│   ├── inspect_dataset.py              # High-performance Polars inspection engine
│   ├── preprocess.py                   # Preprocessing & temporal window partitioning
│   ├── build_graphs.py                 # PyG graph construction & feature scaling
│   ├── train_baseline.py               # Logistic Regression baseline trainer
│   ├── train_world_model.py            # GNN + World Model end-to-end Dual Loss trainer
│   ├── evaluate.py                     # Multi-step benchmark evaluation (World Model vs Baseline)
│   └── run_forecast.py                 # CLI demonstration tool (Forecast vs Actual + MITRE)
├── dashboard/
│   └── app.py                          # Offline Streamlit SOC decision-support dashboard
└── reports/
    └── dataset_inspection.json         # Master dataset inspection report
```

---

## 3. Step-by-Step Execution Guide

All commands should be executed from the project root using PowerShell and the configured virtual environment:

### Step 1: Inspect Raw Dataset
Validates all 8 raw CSV files, handles dirty headers, audits Infinities/NaNs, parses monotonic 24-hour timestamps, and generates topological density metrics:
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/inspect_dataset.py --data-dir data/cic2017 --output reports/dataset_inspection.json
```

### Step 2: Preprocess Flows & Create Temporal Windows
Applies median imputation fitted on **Train only** (zero data leakage) and partitions traffic into 30-second windows:
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/preprocess.py --config configs/config.yaml
```
*Outputs: `data/processed/train_windows.pkl`, `val_windows.pkl`, `test_windows.pkl`, and `models/preprocessors.pkl`.*

### Step 3: Construct PyG Graph Windows ($S[t]$)
Builds PyG `Data` graph objects, extracts 9d node and 7d edge features, fits RobustScalers on Train graphs only, and enforces the `MAX_EDGES = 10,000` priority sampling guardrail:
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/build_graphs.py --config configs/config.yaml
```
*Outputs: `data/graphs/train_graphs.pt`, `data/graphs/val_graphs.pt`, and `data/graphs/test_graphs.pt`.*

### Step 4: Train Baseline Model
Trains a Logistic Regression benchmark using flat window summary statistics:
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/train_baseline.py --config configs/config.yaml
```
*Outputs: `models/baseline_model.pkl`.*

### Step 5: Train GNN Encoder + GRU World Model
Trains the end-to-end AI World Model on your NVIDIA RTX 4060 GPU using mixed precision (`torch.amp`) and the Dual Loss:
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/train_world_model.py --config configs/config.yaml --epochs 20 --batch-size 32
```
*Outputs: Best model checkpoint saved to `models/world_model.pt`.*

### Step 6: Multi-Step Evaluation Benchmark
Evaluates the World Model against the Baseline on the out-of-time Test set across horizons $t+1 \dots t+5$:
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/evaluate.py --config configs/config.yaml --output reports/evaluation_report.json
```
*Outputs: Comparative metrics table (Macro F1, Precision, Recall, False Positive Rate, Lead Time).*

### Step 7: Run CLI Forecasting Demonstration
Simulates a live forecast scenario showing Model Forecast vs Actual Future, MITRE ATT&CK tactics, and gradient feature attributions:
```powershell
& "d:\sih_project\.venv\Scripts\python.exe" scripts/run_forecast.py --config configs/config.yaml --window-idx 20 --horizon 5
```

### Step 8: Launch Streamlit SOC Decision Support Dashboard
Launches the offline Cybersecurity Operations Center command dashboard:
```powershell
& "d:\sih_project\.venv\Scripts\streamlit.exe" run dashboard/app.py
```
Open your browser at `http://localhost:8501` to view:
- Current Network State & Threat Probability
- Interactive Topology Subgraph (NetworkX / Plotly)
- Multi-step Forecast vs Actual Trajectory Progression
- MITRE ATT&CK Threat Intelligence & Recommended Incident Response Actions
- Real Gradient Feature Attribution & Temporal Influence Curve ($t-N \dots t$).

---

## 4. Hardware & VRAM Guardrails
- **GPU**: Optimized for NVIDIA RTX 4060 (8 GB VRAM) with automatic `torch.amp` float16 mixed precision.
- **Fixed Latent Dimension**: Regardless of host count ($|V|$), latent state $z[t]$ is strictly **128-dimensional** (64d mean pool + 64d max pool).
- **Edge Guardrail**: If edge count exceeds 10,000, priority sampling deterministically retains high-volume, high-SYN, and diverse communication channels without random dropping.
