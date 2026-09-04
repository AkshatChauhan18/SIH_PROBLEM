# SENTINEL-X: LSTM World Model & Pipeline Guide
**SIH Problem Statement SIH26153 | AI Based Network Attack Forecasting**

> *"Traditional IDS classifies isolated network flows. Sentinel-X models network behaviour as an evolving trajectory."*

---

## 1. The Core Idea (Explain in 15 Seconds)

An attack is **not an isolated event**—it is a sequence:
$$\text{Port Scanning (Reconnaissance)} \longrightarrow \text{Credential Probing (Initial Access)} \longrightarrow \text{High-Volume Payload (Impact)}$$

Traditional IDS only sounds an alarm when the heavy malicious payload arrives.  
**Sentinel-X uses an LSTM World Model to learn the momentum of network states and simulate where the attack trajectory will be 25 seconds from now.**

---

## 2. The 4-Step Pipeline

```
[ Step 1: Ingest ]
Raw Network Traffic
       │
       ▼ (Group into 5-Second Windows)
[ Step 2: State Space ]
Temporal Network State S(t)  [28 Features]
  • Volume: Flows, Packets, Bytes, Flow Rates
  • TCP Flags: SYN Ratio, ACK Ratio, RST Ratio
  • Topology: Unique Ports, Port Diversity (ports/flow)
  • Jitter: Flow IAT Mean & Std Dev
       │
       ▼ (Stack Past 10 Windows = 50 Seconds of History)
[ Step 3: Deep Learning Core ]
Sequence: [ S(t-9), S(t-8), ..., S(t) ]
       │
       ▼
┌────────────────────────────────────────┐
│      PyTorch LSTM (2 Layers, 128 Hidden)│
│               (RTX 4060 GPU)           │
└────────────────────────────────────────┘
       │
       ├──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
 Head 1: World Model        Head 2: Threat Gate        Head 3: Stage
 Next State S(t+1)          Attack Probability         MITRE ATT&CK Stage
 [Continuous Dynamics]      [ 0% to 100% ]             [Recon, Access, Impact]
       │
       ▼
[ Step 4: Forward Rollout Simulation ]
Loop: Feed predicted S(t+1) back into the LSTM to simulate S(t+2) ➔ S(t+3) ➔ S(t+4) ➔ S(t+5)
       │
       ▼
Result: Attack Probability Curve (NOW ➔ +1 ➔ +2 ➔ +3 ➔ +4 ➔ +5)
        + 20-Second Advance Warning Alert Before Attack Peaks!
```

---

## 3. What is Inside the LSTM Model?

File: `src/model.py` (`NetworkWorldModel`)

* **Input:** A sequence of 10 historical states: shape `(Batch, 10, 28)`.
* **LSTM Layers:** 2 stacked LSTM layers with hidden dimension `128` and dropout `0.2`.
  * Learns temporal dependencies (e.g., how an increase in port diversity at $t-3$ followed by SYN flag spikes at $t-1$ leads to an attack at $t+2$).
* **Shared Representation:** Dense layer (`128 ➔ 64`) with Layer Normalization and ReLU.
* **3 Multi-Task Heads:**
  1. **Next-State Head (`64 ➔ 28`):** Predicts the continuous physical features of the next window $\hat{S}_{t+1}$. This makes it a **"World Model"** because it learns how the network environment itself changes over time.
  2. **Attack Probability Head (`64 ➔ 1` + Sigmoid):** Forecasts whether the next window is malicious ($0.0$ to $1.0$).
  3. **Stage Head (`64 ➔ 6` + Softmax):** Classifies the attack stage among:
     * `Normal` (Benign)
     * `Reconnaissance` (PortScan)
     * `Initial Access` (Brute Force, Patator, Web Attack)
     * `Lateral Movement` (Infiltration)
     * `Command & Control` (Bot)
     * `Impact` (DDoS, DoS)

---

## 4. How Forward Rollout Works (Auto-Regression)

To look 5 windows ($25\text{ seconds}$) into the future:

1. Give the LSTM windows $t-9$ through $t$.
2. The LSTM predicts state $\hat{S}(t+1)$, probability $P(t+1)$, and stage.
3. Drop the oldest window ($t-9$) and append the newly predicted state $\hat{S}(t+1)$.
4. Run the LSTM again on $[t-8, \dots, \hat{S}(t+1)]$ to predict $\hat{S}(t+2)$.
5. Repeat up to step $+5$.

**Output:**
```
NOW (t)  ──► Attack Probability:  2.1%  |  Stage: Normal
 +1      ──► Attack Probability:  8.5%  |  Stage: Normal
 +2      ──► Attack Probability: 39.2%  |  Stage: Reconnaissance
 +3      ──► Attack Probability: 72.4%  |  Stage: Reconnaissance  <-- [!] WARNING TRIGGERED
 +4      ──► Attack Probability: 89.5%  |  Stage: Initial Access
 +5      ──► Attack Probability: 95.2%  |  Stage: Initial Access
```

---

## 5. Why This Beats Traditional IDS (The Judge Punchline)

| Feature | Traditional IDS (e.g. Snort / Suricata / Random Forest) | SENTINEL-X (LSTM World Model) |
|:---|:---|:---|
| **What it examines** | Single, isolated network packet or flow | Temporal trajectory over the past 50 seconds |
| **When it alerts** | **After** attack payload enters the network | **20 seconds BEFORE** the attack reaches full intensity |
| **Lead Time** | **0.0 seconds** | **+15 to +25 seconds** |
| **Actionability** | Reactive incident cleanup | Proactive automated firewall rule injection |
