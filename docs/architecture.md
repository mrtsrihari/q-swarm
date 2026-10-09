# Q-Swarm RL: Architecture & Implementation Plan

> **Hybrid Quantum-Reinforcement Learning for Real-Time Swarm Pathfinding and Dynamic Mesh Optimization in GPS-Denied Emergency Environments**
> Q-Hack India 2026 | Quantum AI & Software Track

---

## 1. System Overview

Q-Swarm RL is a local-first, reproducible multi-agent simulation and policy evaluation platform. It explores whether a low-qubit Variational Quantum Circuit (VQC) can function as a compact, nonlinear decision core for multi-drone pathfinding under GPS denial, obstacle hazards, and ad-hoc communication mesh constraints.

```
+-----------------------------------------------------------------------------------+
|                                  Dashboard & API                                  |
|   - Streamlit Multi-view UI (Live Sim, Mesh Graph, Quantum Inspector, Benchmark)  |
|   - FastAPI Simulation & Inference Service                                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                 Evaluation Layer                                  |
|   - Heuristic Baseline        - Compact Classical DQN                             |
|   - Parameter-Matched MLP     - Hybrid VQC Policy (4 Qubits, Angle + CZ/CNOT)     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              State & Feature Pipeline                             |
|   - Local GPS-Denied Sensing (Odometry + Noise)                                   |
|   - Permutation-Invariant Neighbor Summaries                                      |
|   - Graph Metrics (Laplacian spectrum proxy, degree, link quality)                |
|   - Feature Compressor -> Normalization / Bounded [-pi, pi]                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         Swarm Simulation Environment                              |
|   - Continuous / Discrete 2D Arena (4 drones default; 2, 4, 6, 8 configurable)    |
|   - Static Obstacles & Dynamic Obstacle Hazards                                   |
|   - Target / Rescue Search Coverage Model                                         |
|   - Dynamic Mesh Network (Distance + Line-of-Sight Link Quality via NetworkX)     |
|   - Multi-objective Transparent Reward Function                                   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Subsystems

### 2.1 Swarm Simulation Environment (`src/qswarm/envs/`)
- **`swarm_env.py`**: Gymnasium-compliant multi-agent environment orchestrating step updates, collision checks, coverage grids, and terminal conditions.
- **`dynamics.py`**: 2D kinematics with discrete actions: `{STAY, NORTH, SOUTH, EAST, WEST}` (plus optional diagonals/speed modes). Battery discharge proportional to velocity and radio activity.
- **`obstacles.py`**: Static polygon/box obstacles and dynamic moving hazards (patrol or linear bounce vectors).
- **`sensing.py`**: GPS-denied simulation: noisy local odometry, limited radar/lidar sensing radius, relative distance to rescue targets.
- **`communications.py`**: Ad-hoc mesh network via NetworkX. Node per drone, edge weights based on distance and LoS obstruction. Metrics: connected components, largest component ratio, algebraic connectivity (Fiedler value proxy), average degree.
- **`rewards.py`**: Multi-objective reward balancing target progress, coverage gain, mesh connectivity, collision avoidance, and energy consumption.

### 2.2 Feature Pipeline & Permutation Invariance (`src/qswarm/policies/features.py`)
Each drone observes:
1. Self-state: battery level, local velocity, target bearing/distance (noisy).
2. Obstacle proximity: 4-quadrant distance radar.
3. Swarm & Mesh context: local degree, mean link quality to 1-hop neighbors, distance to swarm centroid, connectivity risk flag.
Compressed into a fixed-size vector (e.g., 6 to 8 dimensions), scaled strictly to $[-\pi, \pi]$ for angle encoding.

### 2.3 Policy Architectures (`src/qswarm/policies/`)
1. **Heuristic Baseline**: Potential-field controller combining target attraction, obstacle repulsion, and mesh cohesion springs.
2. **Compact Classical MLP**: 2-layer MLP (e.g., 64 -> 32 -> 5 actions).
3. **Parameter-Matched Classical Baseline**: Scaled down MLP matching the exact trainable parameter budget of the hybrid model (~20-40 parameters).
4. **Hybrid VQC Policy**:
   - 4-qubit circuit with angle embedding ($R_y / R_z$).
   - Strongly entangling / hardware-efficient ansatz layers (CNOT/CZ rings + trainable parameterized single-qubit rotations).
   - Pauli-$Z$ expectation measurements: $\langle Z_i \rangle \in [-1, 1]$.
   - Linear projection head: 4 expectations $\to$ 5 action logits/Q-values.
   - Differentiable simulator training via parameter-shift rule / statevector gradients, plus noisy simulator evaluation.

### 2.4 Training, Benchmarking & Reproducibility (`src/qswarm/training/`, `src/qswarm/experiments/`)
- Deterministic seeding across NumPy, PyTorch, Python random, and Qiskit.
- Config-driven runs via YAML (`configs/`).
- Benchmark runner comparing wall-clock training time, per-step inference latency, success rate, coverage ratio, collision rate, and connected component stability over identical seeds.
- Automated report generator producing Markdown summaries and charts from raw JSON/CSV data.

### 2.5 Presentation Layer (`src/qswarm/dashboard/`, `src/qswarm/api/`)
- **FastAPI**: Endpoints for running simulation steps, evaluating checkpoints, and exporting telemetry.
- **Streamlit Dashboard**:
  - Live 2D arena visualizer with drone trails, obstacle masks, target zones, and communication mesh edges.
  - Mesh metrics graph and real-time connectivity telemetry.
  - Quantum inspector displaying state vector angles, circuit schematic, gate depth, and qubit expectations.
  - Experiment comparison tables with raw data export.
  - Scientific honesty & limitations page.

---

## 3. Milestones & Delivery Roadmap

- **Milestone 0**: Monorepo skeleton, dependencies, configs, typing, Ruff, Pytest setup.
- **Milestone 1**: 2D simulation environment, NetworkX mesh graph, multi-objective rewards, heuristic baseline, trajectory visualizer.
- **Milestone 2**: Compact classical DQN/actor-critic baseline, parameter-matched classical baseline, training & evaluation loops.
- **Milestone 3**: Hybrid VQC policy with Qiskit, angle encoding, ansatz, parameter-shift / automatic differentiation, noisy simulator path, hardware smoke test.
- **Milestone 4**: Benchmark suite across 5 standardized scenarios (`open_field`, `rescue_maze`, `dynamic_hazard`, `mesh_tradeoff`, `scale_test`), raw data logging, chart/report export.
- **Milestone 5**: Streamlit dashboard, FastAPI service, end-to-end integration tests, documentation, and hackathon presentation demo script.
