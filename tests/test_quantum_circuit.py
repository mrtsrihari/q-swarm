"""Tests for Qiskit VQC circuit, expectations, parameter-shift gradients, and resources."""

import numpy as np
import torch

from qswarm.policies.hybrid_vqc import (
    HybridVQCPolicy,
    VQCExecutionEngine,
    build_vqc_circuit,
)


def test_build_vqc_circuit_structure():
    qc, in_params, var_params = build_vqc_circuit(num_qubits=4, num_layers=2, entanglement="ring")
    assert qc.num_qubits == 4
    assert len(in_params) == 4
    assert len(var_params) == 16
    assert qc.depth() > 0


def test_build_vqc_circuit_disentangled():
    qc, in_params, var_params = build_vqc_circuit(num_qubits=4, num_layers=2, entanglement="none")
    assert qc.num_qubits == 4
    # Zero CX gates in disentangled product state
    assert dict(qc.count_ops()).get("cx", 0) == 0


def test_vqc_execution_expectations():
    engine = VQCExecutionEngine(num_qubits=4, num_layers=1)
    x = np.array([0.5, -0.3, 1.2, -0.8], dtype=np.float32)
    weights = np.zeros(engine.num_weights, dtype=np.float32)

    exp_vals = engine.evaluate_expectations(x, weights)
    assert exp_vals.shape == (4,)
    assert np.all(exp_vals >= -1.0) and np.all(exp_vals <= 1.0)


def test_parameter_shift_gradient_finite_difference():
    """Verify that parameter shift rule matches finite difference approximation."""
    engine = VQCExecutionEngine(num_qubits=4, num_layers=1)
    x = np.array([0.2, 0.4, -0.5, 0.1], dtype=np.float32)
    weights = np.random.uniform(-0.5, 0.5, size=engine.num_weights).astype(np.float32)

    jacobian_ps = engine.compute_parameter_shift_gradient(x, weights)

    eps = 1e-4
    w_plus = weights.copy()
    w_plus[0] += eps
    w_minus = weights.copy()
    w_minus[0] -= eps
    grad_fd = (engine.evaluate_expectations(x, w_plus)[0] - engine.evaluate_expectations(x, w_minus)[0]) / (2 * eps)

    assert np.isclose(jacobian_ps[0, 0], grad_fd, atol=1e-3)


def test_hybrid_vqc_backward_flow():
    """Test PyTorch autograd backward pass through VQC."""
    policy = HybridVQCPolicy(num_qubits=4, num_layers=1, output_dim=5)
    inputs = torch.randn(2, 4, requires_grad=False)

    q_vals = policy(inputs)
    assert q_vals.shape == (2, 5)

    loss = q_vals.sum()
    loss.backward()

    assert policy.quantum_weights.grad is not None
    assert torch.all(torch.isfinite(policy.quantum_weights.grad))
    assert policy.head.weight.grad is not None


def test_resource_estimation():
    policy = HybridVQCPolicy(num_qubits=4, num_layers=2, entanglement="ring")
    res = policy.estimate_resources()

    assert res["num_qubits"] == 4
    assert res["total_trainable_parameters"] == 41
    assert res["circuit_depth_ideal"] > 0
    assert res["circuit_depth_transpiled"] > 0
    assert res["two_qubit_gates_transpiled"] > 0
    assert res["shots"] == 1024


def test_noisy_simulator_mode():
    noise_cfg = {"depolarizing_error": 0.05, "readout_error": 0.02}
    noisy_policy = HybridVQCPolicy(
        num_qubits=4,
        num_layers=1,
        noise_model=noise_cfg,
        use_noise=True,
    )
    obs = np.ones(12, dtype=np.float32) * 0.5
    action = noisy_policy.select_action(obs)
    assert 0 <= action < 5
