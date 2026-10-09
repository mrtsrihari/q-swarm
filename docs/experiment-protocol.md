# Experiment Protocol & Benchmarking Guidelines

---

## 1. Experimental Methodology

All evaluation in **Q-Swarm RL** is strictly deterministic, seeded, and auditable. Every experiment run automatically records:
- Operating system, CPU, RAM, Python version, PyTorch version, Qiskit version.
- Exact random seeds for environment initialization, obstacle trajectories, and policy initialization.
- Full step-level telemetry (positions, link qualities, energy, collisions, coverage).

---

## 2. Standard Benchmark Scenarios

| Scenario | Grid / Arena | Obstacles | Dynamic Hazards | Swarm Size | Objective Emphasis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`open_field`** | $20 \times 20$ | None | None | 4 Drones | Baseline convergence & formation cohesion |
| **`rescue_maze`** | $20 \times 20$ | 8 static barriers | None | 4 Drones | Detour pathfinding & obstacle avoidance |
| **`dynamic_hazard`** | $20 \times 20$ | 4 static | 2 moving hazards | 4 Drones | Real-time reactive avoidance |
| **`mesh_tradeoff`** | $25 \times 25$ | 6 static | None | 4 Drones | Wide targets forcing mesh disconnection risk |
| **`scale_test`** | $30 \times 30$ | 8 static | 2 moving | 6-8 Drones | Scaling multi-agent coordination |

---

## 3. Evaluated Metrics

For every model and scenario, we evaluate:
1. **Success Rate (%)**: Percentage of episodes where all rescue targets are reached before timeout or battery depletion.
2. **Mean Episode Return**: Cumulative discounted reward across all swarm members.
3. **Exploration / Search Coverage (%)**: Proportion of the arena grid tiles visited by at least one drone.
4. **Collision Count / Rate**: Frequency of drone-to-obstacle or drone-to-drone collisions.
5. **Mesh Connected Ratio**: Fraction of time steps where the communication graph has exactly 1 connected component.
6. **Algebraic Connectivity ($\lambda_2$)**: Fiedler eigenvalue of the graph Laplacian, indicating network robustness.
7. **Trainable Parameter Count**: Total learned weights in the policy.
8. **Inference Latency (ms/step)**: Wall-clock time required to generate actions for all 4 drones.
9. **Wall-clock Training Time (s)**: Total training duration.

---

## 4. Reproducibility Commands

To reproduce the benchmark suite:
```bash
# Smoke benchmark (fast verification, 2 seeds)
python scripts/run_benchmark.py --config configs/experiments/benchmark.yaml --smoke

# Full benchmark (statistical significance, 5 seeds)
python scripts/run_benchmark.py --config configs/experiments/benchmark.yaml

# Generate Markdown report and comparison charts
python scripts/export_report.py --results results/latest
```
