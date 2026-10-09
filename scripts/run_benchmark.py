#!/usr/bin/env python3
"""Run reproducible cross-model benchmark across scenarios and seeds."""

import argparse
import sys

from qswarm.experiments.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Q-Swarm RL Benchmark Suite")
    parser.add_argument(
        "--config",
        default="configs/experiments/benchmark.yaml",
        help="Path to benchmark config",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke benchmark with reduced seeds/steps",
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(config_path=args.config, smoke=args.smoke)
    output_dir = runner.run()
    print(f"Benchmark finished successfully. Results stored in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
