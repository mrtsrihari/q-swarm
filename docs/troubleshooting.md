# Troubleshooting & FAQ

---

## 1. Environment & Dependencies

### `ModuleNotFoundError: No module named 'qswarm'`
Ensure you have installed the package in editable mode or added the repository root to `PYTHONPATH`:
```bash
pip install -e .
# Or on Windows PowerShell:
$env:PYTHONPATH = "."
```

### Qiskit Version Compatibility
`Q-Swarm RL` is built on modern Qiskit (`>= 1.0.0` and `2.x`). We use the modern primitives and `qiskit.quantum_info` statevector and operator abstractions. Legacy imports like `qiskit.circuit.QuantumInstance` or deprecated `qiskit.opflow` are not used.

### Torch & CUDA
The simulation and quantum models are optimized to run smoothly on standard CPUs without requiring GPU or CUDA drivers. If PyTorch complains about CUDA on Windows, it will automatically fallback to CPU execution.

---

## 2. Quantum Simulation & Hardware

### "Can I run without an IBM Quantum token?"
**Yes!** The default backend is `"simulator"`, which executes local high-performance statevector evolution. No cloud accounts, tokens, or network requests are required.

### "How do I configure IBM Quantum credentials?"
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Paste your API token:
   ```env
   IBM_QUANTUM_TOKEN=your_actual_token_here
   ```
3. Run the hardware smoke test:
   ```bash
   python scripts/hardware_smoke_test.py --backend ibm_hardware
   ```
   If credentials are absent or invalid, the test skips gracefully and validates circuit compilation locally without throwing errors.

---

## 3. Streamlit Dashboard

### Dashboard Port Conflict
If port 8501 is already in use:
```bash
streamlit run src/qswarm/dashboard/app.py --server.port 8502
```
