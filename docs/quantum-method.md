# Quantum Method & VQC Policy Architecture

---

## 1. Hybrid Quantum-Classical Policy Design

In **Q-Swarm RL**, the decision engine for each drone in the swarm is framed as a hybrid variational quantum-classical network. The circuit serves as a compact nonlinear feature interaction kernel mapping compressed swarm states to action-value expectations.

```
       Swarm Observation s ∈ ℝ¹²
                 │
                 ▼
  ┌─────────────────────────────┐
  │ Graph-Aware State Compressor│  Relative Goal, Obstacle Radar, Mesh Degree & Link Quality
  └─────────────────────────────┘
                 │
                 ▼
       Compressed State x ∈ [-π, π]⁴
                 │
                 ▼
  ┌─────────────────────────────┐
  │      Angle Encoding         │  R_y(x_i) on qubit i
  └─────────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │ Parameterized Quantum Layer │  Layer 1: R_y(θ_1) + R_z(θ_2) + Ring Entanglers (CNOT)
  │     (Hardware Efficient)    │  Layer 2: R_y(θ_3) + R_z(θ_4) + Ring Entanglers (CNOT)
  │   [Ablation: 0 CNOT Option] │  (Ablation: Disentangled Product State)
  └─────────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  Expectation Value Readout  │  ⟨Z_i⟩ = ⟨ψ(x,θ)| Z_i |ψ(x,θ)⟩ ∈ [-1, 1]
  │  [Readout Error Mitigation] │  (Mitigated via Confusion Matrix Inversion M⁻¹)
  └─────────────────────────────┘
                 │
                 ▼
       Readout Vector z ∈ ℝ⁴
                 │
                 ▼
  ┌─────────────────────────────┐
  │    Classical Action Head    │  Q(s, a) = W · z + b   (W ∈ ℝ⁵ˣ⁴, b ∈ ℝ⁵)
  └─────────────────────────────┘
                 │
                 ▼
      Action Values Q ∈ ℝ⁵ (STAY, NORTH, SOUTH, EAST, WEST)
```

---

## 2. Quantum Graph-Aware Encoding vs Simple Angle Encoding

We formulate and empirically compare two encoding schemes:

### 2.1 Graph-Aware Feature Encoding (Proposed)
Synthesizes topological swarm metrics into the 4 quantum rotation angles:
1. **$x_0$ (Target Bearing)**: Directional navigation angle:
   $$x_0 = \text{arctan2}(\Delta y_{\text{target}}, \Delta x_{\text{target}}) \in [-\pi, \pi]$$
2. **$x_1$ (Obstacle Clearance)**: Normalized minimum obstacle clearance:
   $$x_1 = \pi \cdot \min_{k} \text{radar}_k \in [-\pi, \pi]$$
3. **$x_2$ (Graph Mesh Degree & Connectivity Risk)**: Local degree normalized in graph:
   $$x_2 = \pi \cdot \left(\frac{2 \cdot \text{deg}(i)}{N - 1} - 1\right) \in [-\pi, \pi]$$
4. **$x_3$ (Energy & Kinetic State)**: Battery percentage and speed tension:
   $$x_3 = \pi \cdot (0.7 \cdot \text{bat} - 0.3 \cdot \|\vec{v}\|) \in [-\pi, \pi]$$

### 2.2 Simple Angle Baseline (Ablation Control)
Encodes purely kinematic coordinates without graph awareness:
$$x_{\text{simple}} = \pi \cdot [x_{\text{norm}}, y_{\text{norm}}, \text{arctan2}(v_y, v_x), \min \text{radar}]$$

*Ablation Result*: Ablating graph features leads to severe network fragmentation in maze environments, proving that graph features directly drive mesh preservation.

---

## 3. Variational Ansatz & Entanglement Ablation

We implement two ansatz topologies to make the contribution of entanglement testable:

### 3.1 Entangled Ring Ansatz
Alternates parameterized single-qubit rotations with nearest-neighbor ring CNOT gates:
$$U_{\text{ent}} = \prod_{k=0}^{3} \text{CNOT}(k, (k+1) \bmod 4)$$
Creates multi-qubit correlations between navigation, obstacle, and mesh state dimensions.

### 3.2 Disentangled Product State (Ablation Baseline)
Removes all entangling gates ($CX = 0$):
$$U_{\text{disentangled}}(\vec{\theta}) = \bigotimes_{k=0}^{3} \left( R_z(\theta_{k, 2}) R_y(\theta_{k, 1}) \right)$$
Evaluates whether a separable single-qubit rotation model suffices.

---

## 4. Gradients and Optimization

Gradients with respect to variational parameters $\theta_j$ are computed via the **Parameter-Shift Rule**:
$$\frac{\partial \langle Z_k \rangle}{\partial \theta_j} = \frac{1}{2} \left( \langle Z_k \rangle_{\theta_j + \frac{\pi}{2}} - \langle Z_k \rangle_{\theta_j - \frac{\pi}{2}} \right)$$
integrated seamlessly into PyTorch autograd via a custom `torch.autograd.Function`.

---

## 5. Readout Error Mitigation

On noisy backends and physical quantum hardware, measurement bit-flips perturb the raw expectation values:
$$\langle Z \rangle_{\text{raw}} = (1 - 2 p_{\text{ro}}) \langle Z \rangle_{\text{ideal}}$$
We implement 1-qubit confusion matrix inversion:
$$\langle Z \rangle_{\text{mitigated}} = \frac{\langle Z \rangle_{\text{raw}}}{1 - 2 p_{\text{ro}}}$$
clamped to $[-1, 1]$.

---

## 6. Quantum Optimization Extension: QAOA for Mesh Relay Selection

When a swarm communication graph fragments into disconnected components, selecting the optimal drone nodes to act as high-power bridge relays is framed as a graph cut optimization subproblem:

### 6.1 Problem Formulation
Given candidate bridge graph $G = (V, E)$, find binary assignment $z \in \{0, 1\}^N$:
$$\max_{z} \sum_{(u, v) \in E} w_{uv} \frac{1 - Z_u Z_v}{2} - \lambda \sum_{u} Z_u$$

### 6.2 QAOA Circuit
1. Uniform superposition $|+\rangle^{\otimes N} = H^{\otimes N} |0\rangle^{\otimes N}$.
2. Alternating layers of cost Hamiltonian evolution $e^{-i \gamma H_C}$ (via $CX \cdot R_z(2\gamma w) \cdot CX$) and mixer evolution $e^{-i \beta H_M}$ (via $R_x(2\beta)$).
3. Parameter optimization via COBYLA.
4. Compared against exact brute-force search and greedy heuristics.
