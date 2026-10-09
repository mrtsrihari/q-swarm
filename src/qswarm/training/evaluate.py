"""Policy evaluation module across reproducible scenarios and seeds."""

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np

from qswarm.envs.swarm_env import SwarmEnv
from qswarm.policies.classical import ClassicalMatchedPolicy, ClassicalMLPPolicy
from qswarm.policies.heuristic import HeuristicSwarmPolicy


def evaluate_policy(
    policy_type: str = "classical",
    checkpoint: Optional[str] = None,
    scenario: str = "rescue_maze",
    seeds: Optional[List[int]] = None,
    max_steps: int = 60,
    num_drones: int = 4,
    encoding: str = "graph_aware",
    verbose: bool = True,
) -> Dict[str, any]:
    """Deterministic evaluation of a policy over specified seeds."""
    if seeds is None:
        seeds = [42, 101, 202]

    # Instantiate policy
    loaded_policy = None
    if policy_type == "heuristic":
        loaded_policy = HeuristicSwarmPolicy()
    elif policy_type == "classical":
        if checkpoint and os.path.exists(checkpoint):
            loaded_policy = ClassicalMLPPolicy.load(checkpoint)
        else:
            loaded_policy = ClassicalMLPPolicy()
    elif policy_type == "classical_matched":
        if checkpoint and os.path.exists(checkpoint):
            loaded_policy = ClassicalMatchedPolicy.load(checkpoint)
        else:
            loaded_policy = ClassicalMatchedPolicy()
    elif policy_type in ["hybrid_vqc", "hybrid_vqc_entangled"]:
        from qswarm.policies.hybrid_vqc import HybridVQCPolicy
        if checkpoint and os.path.exists(checkpoint):
            loaded_policy = HybridVQCPolicy.load(checkpoint)
        else:
            loaded_policy = HybridVQCPolicy(entanglement="ring", encoding=encoding)
    elif policy_type == "hybrid_vqc_no_entanglement":
        from qswarm.policies.hybrid_vqc import HybridVQCPolicy
        loaded_policy = HybridVQCPolicy(entanglement="none", encoding=encoding)
    elif policy_type == "hybrid_vqc_noisy":
        from qswarm.policies.hybrid_vqc import HybridVQCPolicy
        noise_cfg = {"depolarizing_error": 0.02, "readout_error": 0.03}
        loaded_policy = HybridVQCPolicy(entanglement="ring", encoding=encoding, noise_model=noise_cfg, use_noise=True)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    returns = []
    successes = []
    coverages = []
    collisions = []
    mesh_connected_ratios = []
    step_latencies_ms = []

    last_trajectory = []

    for s_idx, seed in enumerate(seeds):
        env = SwarmEnv(
            scenario=scenario,
            num_drones=num_drones,
            max_steps=max_steps,
        )
        obs, info = env.reset(seed=seed)
        ep_return = 0.0
        step_count = 0
        connected_steps = 0

        done = False
        while not done:
            t0 = time.perf_counter()

            # Action selection
            if policy_type == "heuristic":
                actions = loaded_policy.select_actions(env)
            else:
                actions = [
                    loaded_policy.select_action(obs[i], epsilon=0.0)
                    for i in range(env.num_drones)
                ]

            latency_ms = (time.perf_counter() - t0) * 1000.0
            step_latencies_ms.append(latency_ms)

            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            ep_return += reward
            step_count += 1
            if info.get("mesh_connected", False):
                connected_steps += 1

        returns.append(ep_return)
        successes.append(1.0 if info.get("success", False) else 0.0)
        coverages.append(info.get("coverage_ratio", 0.0))
        collisions.append(info.get("total_collisions", 0))
        mesh_connected_ratios.append(connected_steps / max(1, step_count))

        if s_idx == len(seeds) - 1:
            last_trajectory = env.trajectory_history

    param_count = (
        loaded_policy.count_parameters()
        if hasattr(loaded_policy, "count_parameters")
        else 0
    )

    summary = {
        "policy": policy_type,
        "scenario": scenario,
        "num_seeds": len(seeds),
        "trainable_parameters": param_count,
        "success_rate": round(float(np.mean(successes)) * 100.0, 1),
        "mean_return": round(float(np.mean(returns)), 2),
        "return_std": round(float(np.std(returns)), 2),
        "mean_coverage": round(float(np.mean(coverages)) * 100.0, 1),
        "mean_collisions": round(float(np.mean(collisions)), 1),
        "mesh_connected_ratio": round(float(np.mean(mesh_connected_ratios)) * 100.0, 1),
        "mean_latency_ms": round(float(np.mean(step_latencies_ms)), 3),
    }

    if verbose:
        print("\n" + "=" * 55)
        print(f"Policy Evaluation Report [{policy_type.upper()} | {scenario}]")
        print("=" * 55)
        for k, v in summary.items():
            print(f"  {k:<25}: {v}")
        print("=" * 55 + "\n")

    return {
        "summary": summary,
        "raw_returns": returns,
        "trajectory": last_trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy")
    parser.add_argument(
        "--policy",
        choices=["heuristic", "classical", "classical_matched", "hybrid_vqc"],
        default="classical",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to policy checkpoint")
    parser.add_argument("--scenario", default="rescue_maze", help="Evaluation scenario")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=60, help="Max steps per episode")

    args = parser.parse_args()
    seeds = [42 + i * 37 for i in range(args.episodes)]

    evaluate_policy(
        policy_type=args.policy,
        checkpoint=args.checkpoint,
        scenario=args.scenario,
        seeds=seeds,
        max_steps=args.max_steps,
        verbose=True,
    )


if __name__ == "__main__":
    main()
