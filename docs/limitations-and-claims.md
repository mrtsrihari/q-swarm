# Scientific Honesty, Known Limitations, and Claims

> **Required Reading for Evaluators, Reviewers, and Hackathon Judges**

---

## 1. Core Scientific Position & Honest Research Hypothesis

Q-Swarm RL is an experimental research prototype investigating the feasibility of compact hybrid Quantum-Reinforcement Learning (QRL) for multi-agent swarm coordination.

> **Formal Research Hypothesis**:  
> *“A shallow graph-informed hybrid VQC may provide a compact policy representation for compressed swarm state, but it may not outperform a well-tuned classical policy in reward, time, scalability, or noise robustness.”*

> **Crucial Disclaimer**:  
> The quantum policy is a small hybrid model evaluated in a simulator / noisy simulator unless hardware evidence is explicitly provided. Results do not demonstrate general quantum advantage, real-time quantum hardware control, or superiority over all classical swarm-control methods.

We explicitly refuse to fabricate or exaggerate metrics. Any comparisons between classical and quantum baselines are reported with exact random seeds, parameter counts, latency benchmarks, and standard deviations.

---

## 2. What This Work Does and Does Not Prove

### What This Work Demonstrates
1. **Representational Feasibility**: A small 4-qubit parameterized quantum circuit (VQC) with 8-16 variational parameters can learn non-trivial multi-drone navigation policies within a compressed feature manifold.
2. **Extreme Parameter Compactness**: The VQC hybrid architecture achieves high parameter efficiency (33-41 trainable parameters total vs. 3,077 in the classical MLP).
3. **Graph-Aware Multi-Agent Coordination**: Integrating swarm graph metrics (local degree, link quality, connectivity risk) into angle encoding directly prevents network partition compared to non-graph spatial baselines.
4. **Empirical Entanglement Ablation**: By benchmarking the VQC with and without entangling gates ($CX$), the role of entanglement is made testable rather than assumed.
5. **Transparent Degradation under Noise**: When subjected to depolarizing and readout error models, policy stability and return degrade predictably, characterizing NISQ susceptibility.

### What This Work DOES NOT Prove
1. **No General "Quantum Advantage"**: We do not claim exponential speedup, polynomial speedup, or higher asymptotic return than state-of-the-art classical RL (PPO, MAPPO, MADDPG).
2. **No Hardware Speedup**: Simulating quantum state vectors on CPU/GPU is computationally slower (~5-7 ms/step) than classical matrix multiplication (~0.5 ms/step). Hardware execution is for inference validation, not low-latency edge flight control.
3. **No Automatic Coordination from Entanglement**: Entanglement within a single agent's decision circuit creates inter-feature correlations for that drone; it does not magically transmit faster-than-light signals across physical drones. Inter-drone communication is governed by explicit RF mesh graphs.
4. **No Direct Flight Deployment**: This simulation operates in a discretized 2D environment with simplified kinematics and simulated GPS denial. It is not an avionics autopilot.

---

## 3. Parameter-Matched Classical Baseline

A common pitfall in quantum ML publications is comparing an 8-parameter quantum circuit against an over-parameterized 10,000-parameter classical MLP, and claiming parameter superiority while ignoring that a tiny classical linear/MLP model could achieve similar performance.

To maintain scientific integrity:
- We compare the Hybrid VQC against both a standard **Compact Classical MLP** (3,077 parameters) AND an explicit **Parameter-Matched Classical Model** constrained to approximately the same parameter budget (**25 parameters**).
- We evaluate wall-clock training time, per-step inference latency, sample efficiency, and task success.

---

## 4. Hardware Evidence & Readout Error Mitigation

- **Ideal Statevector Simulation**: Zero noise, exact statevector expectations via Qiskit. Used for baseline algorithmic verification and training gradients.
- **Calibrated Fake Backend**: `GenericBackendV2` (5-qubit superconducting topology) emulates realistic gate and readout infidelities.
- **Readout Error Mitigation**: Applied via calibration matrix scaling ($M^{-1} p$), recovering expectation values closer to ideal statevector baselines.
- **Physical IBM Quantum Hardware**: Available via opt-in CLI (`scripts/hardware_smoke_test.py`) when valid IBM Quantum API tokens are provided in `.env`. Credentials are never hardcoded or pushed. When credentials are not provided, the system gracefully runs local calibrated emulation without errors.

---

## 5. Quantum Optimization Extension (QAOA)

The QAOA relay drone selection module is an **optimization feasibility experiment** on small graphs ($N \le 6$):
- It demonstrates that a parameterized quantum circuit can identify bridge nodes for repairing disconnected communication graphs.
- It is compared directly against an exact brute-force classical solver and a polynomial-time greedy heuristic.
- We do **not** claim commercial quantum advantage over classical heuristics for these small problem sizes.
