# Q-Swarm RL: Hybrid Quantum-Reinforcement Learning for Real-Time Swarm Pathfinding & Dynamic Mesh Optimization

[![CI](https://github.com/q-swarm-rl/q-swarm/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.2+-purple.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **Q-Hack India 2026 | Quantum AI & Software Track**  
> An empirical, local-first research platform investigating low-qubit Variational Quantum Circuits (VQCs) for multi-agent swarm coordination under GPS denial, obstacle hazards, and ad-hoc communication mesh constraints.

---

## 1. Honest Scientific Framing & Core Hypothesis

> **Formal Research Hypothesis**:  
> *“A shallow graph-informed hybrid VQC may provide a compact policy representation for compressed swarm state, but it may not outperform a well-tuned classical policy in reward, time, scalability, or noise robustness.”*

We explicitly refuse to fabricate or exaggerate metrics:
- **No Unfounded "Quantum Advantage" Claims**: Results do not claim general quantum superiority over all classical RL algorithms.
- **Simulator Latency Reality**: Simulating quantum circuits on classical CPUs takes ~5-7 ms/step compared to ~0.5 ms/step for classical matrix multiplication. Hardware execution is an inference feasibility study, not a real-time flight controller.
- **Fair Controls**: We compare the 33-41 parameter Hybrid VQC against both an over-parameterized Classical MLP (3,077 params) AND an explicit **Parameter-Matched Classical Baseline** (25 params).

---

## 2. System Architecture

![Architecture](assets/architecture.svg)

```text
Simulation Environment (Gymnasium)
  ├─ Kinematics & 5 Discrete Actions (STAY, NORTH, SOUTH, EAST, WEST)
  ├─ Static Obstacles & Dynamic Moving Hazards
  ├─ Simulated GPS Denial (Drift + Gaussian Noise)
  ├─ Dynamic Communication Mesh (NetworkX, Laplacian λ₂ Fiedler Value)
  └─ Transparent Multi-Objective Reward Function

Quantum Graph-Aware State Encoding
  ├─ Synthesizes Goal Vector, Obstacle Clearance, Local Degree, & Link Quality
  └─ Angle Mapping x ∈ [-π, π]⁴ (compared against non-graph Simple Angle baseline)

Decision Engines & Quantum Ablations
  ├─ Deterministic Potential-Field Heuristic (0 params)
  ├─ Parameter-Matched Classical Baseline (25 params)
  ├─ Standard Classical MLP DQN (3,077 params)
  ├─ Hybrid VQC Disentangled (0 CNOTs, separable product state)
  ├─ Hybrid VQC Entangled (Ring CNOTs, 41 params)
  └─ Hybrid VQC Noisy (Calibrated depolarizing & readout noise)

Quantum Optimization Extension
  └─ QAOA Relay Selection for repairing partitioned communication meshes
```

---

## 3. Quickstart: Installation

### Prerequisites
- Python 3.10 or 3.11
- Git

### Copy-Paste Installation

#### Windows (PowerShell)
```powershell
git clone https://github.com/q-swarm-rl/q-swarm.git
cd q-swarm
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m qswarm validate
```

#### macOS / Linux
```bash
git clone https://github.com/q-swarm-rl/q-swarm.git
cd q-swarm
python3 -m pip install --upgrade pip
pip install -e ".[dev]"
python3 -m qswarm validate
```

---

## 4. Run in 3 Commands

### 1. One-Command Heuristic Demo
Run an immediate 4-drone rescue scenario and save trajectory data:
```bash
python scripts/run_demo.py --policy heuristic --scenario rescue_maze --seed 42
```

### 2. One-Command Streamlit Dashboard
Launch the full-stack visualizer with 2D playback, mesh analytics, quantum inspector, empirical benchmarks, and live QAOA trials:
```bash
streamlit run src/qswarm/dashboard/app.py
```
*Or via CLI shortcut:* `python -m qswarm dashboard`

### 3. One-Command Smoke Benchmark & Ablation Report
Run an automated empirical benchmark comparing all models, ablations, and QAOA across 5 disaster scenarios:
```bash
python scripts/run_benchmark.py --smoke
python scripts/export_report.py --results results/latest
```

---

## 5. Quantum Depth Upgrades

### 1. Quantum Graph-Aware State Encoding
- **Graph-Aware Encoding**: Injects local node degree, mean 1-hop link quality, and connectivity risk into the quantum state angles.
- **Ablation Finding**: In maze scenarios, graph-aware encoding achieves **100.0% mesh connected ratio** vs. **30.0%** for the simple angle baseline, confirming that quantum state graph features actively prevent communication partition.

### 2. Full Quantum Ablation Study
We compare:
- Classical MLP DQN (3,077 params)
- Parameter-Matched Classical (25 params)
- Hybrid VQC Disentangled (41 params, 0 CNOTs)
- Hybrid VQC Entangled (41 params, Ring CNOTs)
- Hybrid VQC Noisy (depolarizing + readout errors)

### 3. Hardware Evidence & Readout Error Mitigation
Test execution on superconducting device topologies (`GenericBackendV2` and IBM Quantum Cloud Runtime):
```bash
python scripts/hardware_smoke_test.py --backend simulator
```
To submit inference to an authenticated IBM Quantum cloud QPU:
1. Copy `.env.example` to `.env`: `cp .env.example .env`
2. Set `IBM_QUANTUM_TOKEN=<your_token>`
3. Run: `python scripts/hardware_smoke_test.py --backend ibm_hardware`
*(If no credentials are provided, the script safely falls back to local calibrated hardware emulation.)*

Applies **Readout Error Mitigation** ($M^{-1} p$) to recover expectation values closer to ideal statevector baselines.

### 4. Noise Resilience
Benchmarked across ideal statevector, low noise (0.5%), medium calibrated noise (2%), and severe decoherence (5%). Evaluates action agreement %, return degradation, and diagnoses failure modes.

### 5. Quantum Resource Estimation
Exhaustively reports qubit count (4), variational parameters (16), classical head weights (25), transpiled depth (62), 2-qubit gates (14 ECR), shots (1024), and circuit evaluations per gradient update (32).

### 6. QAOA Mesh Relay Selection Optimization
When drone movement fragments the communication mesh, selecting optimal bridge drones to deploy as relays is framed as a graph cut optimization problem. Evaluates QAOA against Exact Brute-Force and Greedy heuristics:
- QAOA Approximation Ratio: **1.000 (Optimal)**
- Runtime: 19.9 ms (Depth 11, 6 CNOTs)

---

## 6. Training & Evaluation Workflow

### Train Baselines
```bash
# 1. Classical MLP DQN (Smoke run on CPU)
python -m qswarm.training.train_classical --config configs/model/classical.yaml --smoke

# 2. Parameter-Matched Classical Model (~25 params)
python -m qswarm.training.train_classical --config configs/model/classical_matched.yaml --matched --smoke

# 3. Hybrid VQC Policy (4 Qubits, parameter-shift gradients)
python -m qswarm.training.train_hybrid --config configs/model/hybrid_vqc.yaml --smoke
```

### Evaluate Checkpoints
```bash
python -m qswarm.training.evaluate --policy classical --checkpoint results/checkpoints/classical/checkpoint.pt --episodes 10
python -m qswarm.training.evaluate --policy hybrid_vqc --checkpoint results/checkpoints/hybrid_vqc/checkpoint.pt --episodes 5
```

---

## 7. Automated Testing & Verification

Run the full automated test suite:
```bash
python -m pytest -v
ruff check .
```

**35 out of 35 unit/e2e tests pass cleanly** in ~6 seconds across 11 test modules:
- Determinism under fixed seeds
- Kinematic bounds & collision detection
- Dynamic obstacle boundary bouncing
- NetworkX graph Laplacian algebraic connectivity ($\lambda_2$)
- Multi-objective reward component signs & penalties
- Quantum state encodings & feature breakdown
- Parameter-shift gradients & PyTorch autograd flow
- Quantum resource estimation
- Readout error mitigation & noise regimes
- QAOA relay repair optimization & greedy/exact solvers
- FastAPI endpoints & health status.

---

## 8. Immediate Free Deployment on Streamlit Community Cloud

The entire Q-Swarm RL dashboard is prepared for zero-configuration, 1-click deployment on [Streamlit Community Cloud](https://share.streamlit.io/):

### Cloud Deployment Steps

1. **Fork or Push** this repository to your GitHub account:
   ```bash
   git remote add origin https://github.com/mrtsrihari/q-swarm.git
   git push -u origin main
   ```
2. **Log into [share.streamlit.io](https://share.streamlit.io/)** with your GitHub account.
3. **Click "New app"** and fill in the deployment configuration:
   - **Repository**: `mrtsrihari/q-swarm` (or your fork)
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py` *(or `app.py`)*
   - **Advanced settings**: Python `3.10`
4. **Click "Deploy!"**

### Public Cloud Architecture & Security Guardrails

- **Zero-Secret Guarantee**: No IBM Quantum API tokens or cloud credentials are stored or needed in the public repository or deployment.
- **Fast CPU Demo Mode**: Uses bundled, lightweight pre-trained model checkpoints in [`results/checkpoints/`](results/checkpoints/) (~20 KB total) for sub-second, responsive episode simulation on cloud CPUs.
- **Physical Hardware Safe Guard**: Direct cloud QPU submission is disabled by default in the public app to avoid quota consumption and queue lag. Full local hardware execution remains available via CLI when `.env` is configured locally.
- **Local Training Separation**: Resource-heavy RL training runs locally via `python -m qswarm.training.train_hybrid`, while the public web app serves live inference, interactive trajectory playback, and QAOA graph repair analysis.

---

## 9. License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
