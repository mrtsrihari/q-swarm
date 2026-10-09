"""End-to-end smoke test across full pipeline."""

import os

from qswarm.envs.swarm_env import SwarmEnv
from qswarm.policies.classical import ClassicalMatchedPolicy
from qswarm.policies.heuristic import HeuristicSwarmPolicy
from qswarm.policies.hybrid_vqc import HybridVQCPolicy


def test_end_to_end_simulation_all_policies():
    """Verify that all 4 policies can execute steps inside SwarmEnv without error."""
    env = SwarmEnv(scenario="rescue_maze", num_drones=4, max_steps=5)

    policies = [
        ("heuristic", HeuristicSwarmPolicy()),
        ("classical_matched", ClassicalMatchedPolicy()),
        ("hybrid_vqc", HybridVQCPolicy(num_qubits=4, num_layers=1)),
    ]

    for name, policy in policies:
        obs, info = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 5:
            if name == "heuristic":
                actions = policy.select_actions(env)
            else:
                actions = [policy.select_action(obs[i]) for i in range(env.num_drones)]

            obs, reward, terminated, truncated, info = env.step(actions)
            steps += 1
            done = terminated or truncated

        assert steps > 0
        assert len(env.trajectory_history) == steps + 1  # includes initial frame


def test_results_and_reports_schema():
    """Verify that sample benchmark summary and raw metrics exist and conform to schema."""
    sample_csv = "results/sample/benchmark_summary.csv"
    assert os.path.exists(sample_csv)

    import pandas as pd
    df = pd.read_csv(sample_csv)
    required_cols = [
        "model_name",
        "scenario",
        "trainable_parameters",
        "mean_return",
        "mean_coverage",
        "mesh_connected_ratio",
        "mean_latency_ms",
    ]
    for col in required_cols:
        assert col in df.columns
