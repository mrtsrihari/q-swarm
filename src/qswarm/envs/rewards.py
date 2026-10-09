"""Multi-objective transparent reward function for swarm navigation and mesh preservation."""

from dataclasses import dataclass
from typing import Dict, Tuple

from qswarm.envs.communications import MeshMetrics


@dataclass
class RewardWeights:
    target_reach: float = 25.0
    target_progress: float = 2.0
    coverage_step: float = 0.5
    mesh_connected_step: float = 1.5
    algebraic_connectivity_weight: float = 1.0
    mesh_disconnected_penalty: float = -3.0
    collision_penalty: float = -30.0
    obstacle_proximity_penalty: float = -1.0
    step_energy_penalty: float = -0.1
    out_of_bounds_penalty: float = -20.0
    timeout_penalty: float = -10.0


class SwarmRewardCalculator:
    """Calculates multi-objective rewards and individual component contributions."""

    def __init__(self, weights: RewardWeights = None):
        self.weights = weights if weights is not None else RewardWeights()

    def compute_reward(
        self,
        targets_reached: int,
        target_dist_delta: float,
        new_cells_covered: int,
        mesh_metrics: MeshMetrics,
        collisions_count: int,
        min_obstacle_distance: float,
        energy_spent: float,
        out_of_bounds: bool,
        is_timeout: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate total reward and decompose into transparent components."""
        components = {}

        # 1. Target Reach & Progress
        components["target_reach"] = float(targets_reached * self.weights.target_reach)
        components["target_progress"] = float(target_dist_delta * self.weights.target_progress)

        # 2. Coverage Gain
        components["coverage"] = float(new_cells_covered * self.weights.coverage_step)

        # 3. Mesh Quality & Connectivity
        if mesh_metrics.is_fully_connected:
            conn_rew = self.weights.mesh_connected_step + (
                self.weights.algebraic_connectivity_weight * mesh_metrics.algebraic_connectivity
            )
            components["mesh_connectivity"] = float(conn_rew)
        else:
            # Penalize fragmentation proportional to number of disconnected components
            penalty = self.weights.mesh_disconnected_penalty * (mesh_metrics.num_components - 1)
            components["mesh_connectivity"] = float(penalty)

        # 4. Collisions
        components["collision"] = float(collisions_count * self.weights.collision_penalty)

        # 5. Obstacle Proximity (< 1.5 distance)
        if min_obstacle_distance < 1.5:
            prox_pen = self.weights.obstacle_proximity_penalty * (1.5 - min_obstacle_distance)
            components["obstacle_proximity"] = float(prox_pen)
        else:
            components["obstacle_proximity"] = 0.0

        # 6. Energy Use
        components["energy"] = float(energy_spent * self.weights.step_energy_penalty)

        # 7. Boundary / Timeout Penalties
        components["out_of_bounds"] = float(self.weights.out_of_bounds_penalty if out_of_bounds else 0.0)
        components["timeout"] = float(self.weights.timeout_penalty if is_timeout else 0.0)

        total_reward = sum(components.values())
        return float(total_reward), components
