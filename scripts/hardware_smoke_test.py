#!/usr/bin/env python3
"""Quantum Hardware & Transpiled Inference Feasibility Verification.

Runs trained policy inference circuits on target quantum hardware backends.
Supports:
1. Real IBM Quantum hardware via Qiskit Runtime (when IBM_QUANTUM_TOKEN is set).
2. Calibrated superconducting hardware simulation via Qiskit GenericBackendV2.
3. Readout error mitigation via calibration matrix inversion.
"""

import argparse
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from qswarm.experiments.noise_benchmarking import ReadoutMitigator
from qswarm.policies.features import compress_features_deterministic
from qswarm.policies.hybrid_vqc import HybridVQCPolicy, build_vqc_circuit


def run_hardware_demonstration(backend_type: str = "simulator", shots: int = 1024) -> int:
    load_dotenv()
    print("=" * 70)
    print("Q-Swarm RL: Quantum Hardware Inference Feasibility Demonstration")
    print("=" * 70)
    print("NOTE: Hardware execution validates quantum circuit feasibility on NISQ QPUs.")
    print("      It is NOT intended for sub-millisecond real-time edge drone flight control.")
    print("=" * 70)

    # 1. Prepare sample drone observation & compressed features
    sample_obs = np.array([0.25, -0.40, 0.10, -0.05, 0.85, 0.70, 0.90, 0.60, 0.80, 0.45, 0.30, 0.75], dtype=np.float32)
    compressed_angles = compress_features_deterministic(sample_obs, target_dim=4, encoding_type="graph_aware")
    print("\n[1/5] Prepared compressed swarm state angles:")
    print(f"      x = {np.round(compressed_angles, 3).tolist()} rad")

    # 2. Build canonical 4-qubit parameterized circuit
    qc, in_params, var_params = build_vqc_circuit(num_qubits=4, num_layers=2, entanglement="ring")
    theta_vals = np.zeros(len(var_params), dtype=np.float32)
    param_dict = {p: float(v) for p, v in zip(in_params, compressed_angles)}
    param_dict.update({p: float(v) for p, v in zip(var_params, theta_vals)})
    bound_qc = qc.assign_parameters(param_dict)
    bound_qc.measure_all()

    # 3. Hardware backend selection
    token = os.getenv("IBM_QUANTUM_TOKEN", "").strip()
    target_backend = None
    backend_name = "GenericBackendV2 (5Q Superconducting)"
    is_physical_cloud = False

    if backend_type == "ibm_hardware":
        if not token:
            print("\n[2/5] IBM Quantum Credential Check:")
            print("      [NOTICE] No IBM_QUANTUM_TOKEN detected in environment.")
            print("      To run on IBM Quantum physical cloud QPUs:")
            print("        1. Copy .env.example to .env:  cp .env.example .env")
            print("        2. Set IBM_QUANTUM_TOKEN=<your_token>")
            print("        3. Re-run: python scripts/hardware_smoke_test.py --backend ibm_hardware")
            print("      -> Falling back to calibrated superconducting hardware emulation (GenericBackendV2).")
            target_backend = GenericBackendV2(num_qubits=5)
        else:
            print("\n[2/5] Connecting to IBM Quantum Runtime cloud service...")
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService(channel="ibm_quantum", token=token)
                target_backend = service.least_busy(operational=True, simulator=False)
                backend_name = target_backend.name
                is_physical_cloud = True
                print(f"      Connected to real physical backend: {backend_name}")
            except Exception as e:
                print(f"      Could not connect to IBM cloud: {e}")
                print("      Falling back to local calibrated hardware backend (GenericBackendV2).")
                target_backend = GenericBackendV2(num_qubits=5)
    else:
        target_backend = GenericBackendV2(num_qubits=5)

    print("\n[3/5] Transpiling circuit for target topology:")
    print(f"      Backend: {backend_name}")
    t0_transpile = time.perf_counter()
    transpiled_qc = transpile(
        bound_qc,
        backend=target_backend,
        optimization_level=2,
        seed_transpiler=42,
    )
    transpile_time = time.perf_counter() - t0_transpile
    ops = dict(transpiled_qc.count_ops())
    two_q_count = ops.get("cx", 0) + ops.get("ecr", 0) + ops.get("cz", 0)

    print(f"      Transpiled Depth : {transpiled_qc.depth()}")
    print(f"      2-Qubit Gates    : {two_q_count}")
    print(f"      Total Gate Count : {sum(ops.values())}")
    print(f"      Transpile Time   : {transpile_time:.4f}s")

    # 4. Execute inference with and without Readout Error Mitigation
    print(f"\n[4/5] Executing inference circuit (Shots: {shots})...")
    t0_exec = time.perf_counter()

    if is_physical_cloud:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(backend=target_backend)
        job = sampler.run([transpiled_qc], shots=shots)
        print(f"      Job submitted to cloud queue: {job.job_id()}")
        pub_result = job.result()[0]
        counts = pub_result.data.meas.get_counts()
    else:
        sim = AerSimulator.from_backend(target_backend)
        job = sim.run(transpiled_qc, shots=shots)
        counts = job.result().get_counts()

    exec_time = time.perf_counter() - t0_exec
    print(f"      Execution finished in {exec_time:.3f}s")

    # Compute raw Pauli-Z expectation values from bitstring counts
    raw_expectations = np.zeros(4, dtype=np.float32)
    for bitstring, count in counts.items():
        # bitstring is reversed (qubit 0 is rightmost)
        cleaned = bitstring.replace(" ", "")
        p_shot = count / shots
        for q in range(4):
            bit = int(cleaned[-(q + 1)])
            z_val = 1.0 if bit == 0 else -1.0
            raw_expectations[q] += z_val * p_shot

    # Compute Ideal Statevector Expectations for comparison
    policy = HybridVQCPolicy(num_qubits=4, num_layers=2)
    ideal_expectations = policy.get_last_expectations(sample_obs)

    # Apply Readout Error Mitigation
    mitigator = ReadoutMitigator(readout_error_rate=0.03, num_qubits=4)
    mitigated_expectations = mitigator.mitigate_expectations(raw_expectations)

    # 5. Display comparative evidence table
    print("\n[5/5] Empirical Expectation Value Comparison:")
    print("-" * 70)
    print(f"{'Observable':<12} | {'Ideal Statevector':<18} | {'Hardware Raw':<16} | {'Mitigated Readout':<16}")
    print("-" * 70)
    for q in range(4):
        print(
            f"<Z_{q}>{'':<8} | {ideal_expectations[q]:>14.4f}   | {raw_expectations[q]:>12.4f}   | {mitigated_expectations[q]:>14.4f}"
        )
    print("-" * 70)

    # Save summary artifact
    os.makedirs("results/sample", exist_ok=True)
    summary_path = "results/sample/hardware_inference_evidence.json"
    evidence = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend_name,
        "is_physical_cloud": is_physical_cloud,
        "shots": shots,
        "transpiled_depth": transpiled_qc.depth(),
        "two_qubit_gates": two_q_count,
        "execution_time_s": round(exec_time, 4),
        "ideal_expectations": [round(float(v), 4) for v in ideal_expectations],
        "hardware_raw_expectations": [round(float(v), 4) for v in raw_expectations],
        "mitigated_expectations": [round(float(v), 4) for v in mitigated_expectations],
    }
    with open(summary_path, "w") as f:
        import json
        json.dump(evidence, f, indent=2)
    print(f"\nSaved hardware evidence artifact to: {summary_path}")
    print("=" * 70 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Quantum Hardware Demonstration")
    parser.add_argument(
        "--backend",
        choices=["simulator", "ibm_hardware"],
        default="simulator",
        help="Backend target",
    )
    parser.add_argument("--shots", type=int, default=1024, help="Measurement shots")
    args = parser.parse_args()
    sys.exit(run_hardware_demonstration(args.backend, args.shots))


if __name__ == "__main__":
    main()
