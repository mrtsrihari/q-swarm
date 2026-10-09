#!/usr/bin/env python3
"""Run standalone multi-drone simulation demo with trajectory export."""

import argparse
import json
import os
import sys

from qswarm.policies.heuristic import run_heuristic_demo


def main():
    parser = argparse.ArgumentParser(description="Q-Swarm RL Simulation Demo Runner")
    parser.add_argument(
        "--policy",
        choices=["heuristic", "classical", "classical_matched", "hybrid_vqc"],
        default="heuristic",
        help="Policy to execute",
    )
    parser.add_argument(
        "--scenario",
        default="rescue_maze",
        choices=["open_field", "rescue_maze", "dynamic_hazard", "mesh_tradeoff", "scale_test"],
        help="Scenario environment",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--steps", type=int, default=60, help="Max steps per episode")
    parser.add_argument("--save-trajectory", action="store_true", default=True, help="Save trajectory json")
    parser.add_argument("--output-dir", default="results/sample", help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.policy == "heuristic":
        result = run_heuristic_demo(
            scenario=args.scenario,
            seed=args.seed,
            max_steps=args.steps,
            verbose=True,
        )
    else:
        # For non-heuristic policies, we can use the evaluation runner
        from qswarm.training.evaluate import evaluate_policy
        print(f"Evaluating {args.policy} on scenario {args.scenario}...")
        result = evaluate_policy(
            policy_type=args.policy,
            scenario=args.scenario,
            seeds=[args.seed],
            max_steps=args.steps,
        )

    if args.save_trajectory and "trajectory" in result:
        outfile = os.path.join(args.output_dir, f"trajectory_{args.scenario}_{args.seed}.json")
        # Format serializable trajectory
        serializable_traj = []
        for frame in result["trajectory"]:
            serializable_traj.append({
                "step": frame["step"],
                "drones": [
                    {
                        "id": d.drone_id,
                        "pos": [round(float(d.position[0]), 3), round(float(d.position[1]), 3)],
                        "battery": round(float(d.battery), 1),
                        "alive": d.is_alive,
                    }
                    for d in frame["drones"]
                ],
                "obstacles": frame["obstacles"],
                "targets": frame["targets"],
                "edges": frame["edges"],
                "coverage_ratio": round(float(frame["coverage_ratio"]), 3),
                "reward": round(float(frame["reward"]), 2),
            })

        with open(outfile, "w") as f:
            json.dump({"summary": result["summary"], "frames": serializable_traj}, f, indent=2)
        print(f"Saved trajectory data to {outfile}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
