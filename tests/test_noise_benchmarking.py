"""Tests for noise benchmarking and readout error mitigation."""

import numpy as np

from qswarm.experiments.noise_benchmarking import ReadoutMitigator, evaluate_noise_regimes


def test_readout_mitigator_scaling():
    mitigator = ReadoutMitigator(readout_error_rate=0.03, num_qubits=4)
    # If ideal expectation is 1.0, under 3% bitflip it would be ~0.94
    noisy = np.array([0.94, -0.94, 0.50, -0.50], dtype=np.float32)
    mitigated = mitigator.mitigate_expectations(noisy)

    assert np.all(mitigated >= -1.0) and np.all(mitigated <= 1.0)
    # Mitigated expectation should have higher magnitude
    assert abs(mitigated[0]) > abs(noisy[0])
    assert abs(mitigated[1]) > abs(noisy[1])


def test_evaluate_noise_regimes_smoke():
    results = evaluate_noise_regimes(max_steps=5, num_episodes=1)
    assert len(results) == 5
    assert results[0].regime_name == "Ideal Statevector"
    assert results[0].action_agreement_pct == 100.0
    for r in results:
        assert r.action_agreement_pct > 0.0
        assert r.mesh_connected_ratio >= 0.0
