# 3-5 Minute Hackathon Presentation Demo Script

> **Q-Hack India 2026 Presentation Guide**

---

## Act I: The Problem (0:00 - 0:45)
- **Visual**: Show Streamlit Dashboard with GPS denial and fragmented drones.
- **Narrative**:
  > "In emergency disaster zones—collapsed buildings, subterranean tunnels, or jamming fields—drones lose GPS and direct satellite communication. To search for survivors, they must maintain an ad-hoc ad-hoc peer-to-peer radio mesh while navigating dynamic debris. If they scatter too far, the mesh fragments and data is lost. If they stay too close, search coverage stalls."

## Act II: The Hybrid Quantum Approach (0:45 - 2:00)
- **Visual**: Switch to the **Quantum Policy Inspector** tab in the Dashboard.
- **Narrative**:
  > "Edge drones have severe compute and weight limitations. We explore whether a tiny 4-qubit Variational Quantum Circuit can serve as a super-compact decision core. We compress multi-drone local observations and graph Laplacian connectivity into 4 angles, pass them through a ring-entangled parameterized circuit, and measure Pauli-Z expectations.
  >
  > Notice the parameter count: only 41 parameters total, compared to 1,000+ in standard deep RL networks. We also implement a parameter-matched classical baseline to ensure a scientifically fair comparison."

## Act III: Live Demonstration (2:00 - 3:30)
- **Visual**: Run **Live Simulation** on `rescue_maze` with Heuristic, Classical, and Hybrid VQC policies.
- **Narrative**:
  > "Watch the drones navigate around obstacles towards rescue targets. The cyan lines represent the active communication mesh. When a drone gets close to the radio cutoff distance, our mesh penalty pulls it back into range.
  >
  > Here in the inspector, you can see real-time quantum expectation values $\langle Z_i \rangle$ shifting as the drone encounters obstacle fields."

## Act IV: Honest Science & Results (3:30 - 4:30)
- **Visual**: Open the **Benchmark & Experiments** tab. Show the real comparative table.
- **Narrative**:
  > "We hold ourselves to rigorous scientific integrity. We do not claim general quantum advantage. In our reproducible benchmarks across 5 scenarios, the hybrid quantum model achieves competitive search coverage and mesh connectivity while maintaining an order-of-magnitude smaller parameter footprint. However, current simulator latency on classical CPUs is slower than pure matrix operations, highlighting the clear roadmap for future edge-QPU co-processors."
