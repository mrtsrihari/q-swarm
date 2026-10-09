"""Benchmark metrics data structures and aggregation."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class ScenarioResult:
    model_name: str
    scenario: str
    num_seeds: int
    trainable_parameters: int
    success_rate: float
    mean_return: float
    return_std: float
    mean_coverage: float
    mean_collisions: float
    mesh_connected_ratio: float
    mean_latency_ms: float
    train_time_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_experiment_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multi-seed runs into mean and standard deviations."""
    if not runs:
        return {}

    returns = [r["return"] for r in runs]
    successes = [1.0 if r["success"] else 0.0 for r in runs]
    coverages = [r["coverage"] for r in runs]
    collisions = [r["collisions"] for r in runs]
    mesh_ratios = [r["mesh_connected_ratio"] for r in runs]
    latencies = [r["latency_ms"] for r in runs]

    return {
        "num_runs": len(runs),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)) * 100.0,
        "mean_coverage": float(np.mean(coverages)) * 100.0,
        "mean_collisions": float(np.mean(collisions)),
        "mesh_connected_ratio": float(np.mean(mesh_ratios)) * 100.0,
        "mean_latency_ms": float(np.mean(latencies)),
    }
