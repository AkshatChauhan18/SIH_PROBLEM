# SENTINEL-X: Demonstration & Presentation Guide
**SIH Problem Statement SIH26153: AI based Network Attack Forecasting from Network Traffic Data**

> *"Don't just detect the attack. Forecast where it's going."*

---

## ⚡ Quick Start (Launch Command)

### Option A: Interactive SOC Dashboard (Streamlit)
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Terminal Attack Forecaster (CLI)
Evaluate any time window directly in your terminal:
```powershell
# 1. Default run (auto-selects attack escalation window)
.\.venv\Scripts\python.exe forecast.py

# 2. Evaluate a specific time window index (e.g. Window 18, 25, 74)
.\.venv\Scripts\python.exe forecast.py --scenario portscan --window 18

# 3. Evaluate by timestamp string (e.g. 03:55)
.\.venv\Scripts\python.exe forecast.py --scenario ddos --time "03:55"

# 4. List all available time windows with their ground truth in the scenario
.\.venv\Scripts\python.exe forecast.py --scenario portscan --list
```

---

## ⏱️ 2-Minute Judge Walkthrough Flow

| Step | Time | Screen / Action | Script / What to Say to the Judges |
|:---|:---|:---|:---|
| **1** | 0:00 – 0:15 | **Page 1: Overview** | *"Current IDS tools wait for a malicious packet or payload to breach the network before raising an alert. Sentinel-X shifts the paradigm: we model network behaviour as a continuous temporal trajectory to forecast where an attack is heading before it strikes."* |
| **2** | 0:15 – 0:35 | **Page 1: Forecast Curve** | Point to the **Attack Probability Forecast** chart and the **MITRE ATT&CK Progression Pipeline**. *"Here, the network is progressing from nominal state to Reconnaissance. Sentinel-X projects probability surging from 2% to 93% across the next 5 windows."* |
| **3** | 0:35 – 1:00 | **Page 2: Attack Forecast** | Click **2. Attack Forecast** in the sidebar. Point to the architecture pipeline (`Observation S(t)` ➔ `LSTM World Model` ➔ `Forward Rollout`). Show the **Forecast Warning Banner** and forward trajectory table (`NOW ➔ +1 ➔ +2 ➔ +3 ➔ +4 ➔ +5`). *"Our PyTorch LSTM World Model simulates future states auto-regressively, yielding a 20-second lead time before critical impact."* |
| **4** | 1:00 – 1:20 | **Page 4: Explainability** | Click **4. Explainability**. Show the perturbation sensitivity attribution bar chart. *"Judges often ask: why is the AI predicting this? Sentinel-X computes feature attributions. Notice that SYN packet ratio and destination port diversity are the primary drivers elevating the attack trajectory."* |
| **5** | 1:20 – 1:40 | **Page 6: Model Performance** | Click **6. Model Performance**. Show the comparison table between Stateless Logistic Regression and Sentinel-X. *"Stateless models have 0.0s lead time because they only classify flows post-facto. Sentinel-X achieves +20.0s early warning lead time with 100% precision on the test split."* |
| **6** | 1:40 – 2:00 | **Page 7: About / Wrap-up** | Conclude with: *"Sentinel-X runs 100% offline on consumer hardware with GPU acceleration, providing actionable, explainable early warnings for automated firewall isolation."* |

---

## 🏗️ Technical Architecture Highlights

```
Network Flow Stream (CIC-IDS2017)
  ↓ (5-Second Aggregation Windows)
Temporal State Space S(t) ∈ R^28
  • Volume: Total flows, packets, bytes, flow rates
  • Topology: Unique source/dest IPs, unique ports, port diversity
  • Protocol Dynamics: SYN, ACK, RST, FIN, PSH ratios
  • Flow Metrics: Flow IAT mean/std/variance, packet size statistics
  ↓
PyTorch LSTM World Model (2 Layers, Hidden Dim: 128, CUDA RTX 4060)
  ↓
Forward Trajectory Rollout S(t+1) ... S(t+K)
  ↓
Multi-Task Predictions:
  1. Attack Probability: P(Attack) ∈ [0, 1]
  2. MITRE ATT&CK Stage: Normal ➔ Reconnaissance ➔ Initial Access ➔ Lateral Movement ➔ Command & Control ➔ Impact
  ↓
Transparent Explainability (Sensitivity Perturbation Analysis)
```

---

## 📁 Repository Structure

```
d:\sih_project\
├── app.py                     # Streamlit Enterprise SOC Dashboard
├── train.py                   # PyTorch GPU Training Pipeline
├── DEMO_GUIDE.md              # Demonstration script and instructions
├── requirements.txt           # Virtualenv dependencies
├── data\cic2017\              # 8 CIC-IDS2017 CSV flow datasets
├── models\
│   ├── world_model.pth        # Trained PyTorch weights
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── metrics.json           # Evaluation benchmarks
│   └── demo_cache\            # Precomputed scenario sequences
└── src\
    ├── config.py              # MITRE mappings, feature lists, hyperparameters
    ├── preprocessing.py       # Flow aggregation & temporal state builder
    ├── model.py               # PyTorch LSTM World Model architecture
    ├── forecast_engine.py     # Forward rollout simulation engine
    ├── explainability.py      # Perturbation-based sensitivity attributions
    ├── baseline.py            # Logistic Regression baseline & evaluation
    └── demo_data.py           # Scenario manager & raw flow loader
```
