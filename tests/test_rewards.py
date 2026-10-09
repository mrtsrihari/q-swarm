"""Tests for reward components, signs, and penalties."""

from qswarm.envs.communications import MeshMetrics
from qswarm.envs.rewards import RewardWeights, SwarmRewardCalculator


def test_reward_component_signs():
    calc = SwarmRewardCalculator(RewardWeights())

    connected_metrics = MeshMetrics(
        num_nodes=4,
        num_edges=6,
        num_components=1,
        largest_component_ratio=1.0,
        average_degree=3.0,
        algebraic_connectivity=2.0,
        is_fully_connected=True,
        mean_link_quality=0.8,
    )

    disconnected_metrics = MeshMetrics(
        num_nodes=4,
        num_edges=2,
        num_components=2,
        largest_component_ratio=0.5,
        average_degree=1.0,
        algebraic_connectivity=0.0,
        is_fully_connected=False,
        mean_link_quality=0.4,
    )

    # 1. Target reached adds positive reward
    rew_target, comps_target = calc.compute_reward(
        targets_reached=1,
        target_dist_delta=1.0,
        new_cells_covered=0,
        mesh_metrics=connected_metrics,
        collisions_count=0,
        min_obstacle_distance=5.0,
        energy_spent=1.0,
        out_of_bounds=False,
        is_timeout=False,
    )
    assert comps_target["target_reach"] > 0
    assert comps_target["target_progress"] > 0

    # 2. Collision incurs strictly negative penalty
    _, comps_collision = calc.compute_reward(
        targets_reached=0,
        target_dist_delta=0.0,
        new_cells_covered=0,
        mesh_metrics=connected_metrics,
        collisions_count=2,
        min_obstacle_distance=5.0,
        energy_spent=1.0,
        out_of_bounds=False,
        is_timeout=False,
    )
    assert comps_collision["collision"] < 0

    # 3. Disconnected mesh incurs negative penalty
    _, comps_disconn = calc.compute_reward(
        targets_reached=0,
        target_dist_delta=0.0,
        new_cells_covered=0,
        mesh_metrics=disconnected_metrics,
        collisions_count=0,
        min_obstacle_distance=5.0,
        energy_spent=1.0,
        out_of_bounds=False,
        is_timeout=False,
    )
    assert comps_disconn["mesh_connectivity"] < 0
