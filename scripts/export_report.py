#!/usr/bin/env python3
"""Export formatted Markdown report and charts from benchmark results."""

import argparse
import os
import sys

from qswarm.experiments.reporting import generate_benchmark_report


def main():
    parser = argparse.ArgumentParser(description="Export Benchmark Report")
    parser.add_argument(
        "--results",
        default="results/latest",
        help="Path to results directory containing benchmark_summary.csv",
    )
    args = parser.parse_args()

    results_dir = args.results
    if results_dir == "results/latest":
        latest_file = os.path.join("results", "latest_dir.txt")
        if os.path.exists(latest_file):
            with open(latest_file, "r") as f:
                results_dir = f.read().strip()
        else:
            # Look for most recent benchmark dir in results/
            subdirs = [
                os.path.join("results", d)
                for d in os.listdir("results")
                if os.path.isdir(os.path.join("results", d)) and d.startswith("benchmark_")
            ]
            if subdirs:
                results_dir = max(subdirs, key=os.path.getmtime)

    if not os.path.exists(results_dir):
        print(f"Error: Results directory does not exist: {results_dir}")
        return 1

    report_path = generate_benchmark_report(results_dir)
    print(f"Report exported successfully to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
