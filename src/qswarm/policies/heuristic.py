"""Deterministic potential-field heuristic baseline controller."""

from typing import List

import numpy as np

from qswarm.envs.dynamics import ACTION_DELTAS, DroneAction
from qswarm.envs.swarm_env import SwarmEnv


class HeuristicSwarmPolicy:
    """Combines target attraction, obstacle repulsion, and mesh cohesion springs."""

    def __init__(
        self,
        target_attraction_weight: float = 2.0,
        obstacle_repulsion_weight: float = 3.5,
        mesh_cohesion_weight: float = 1.2,
        comm_threshold_ratio: float = 0.75,
    ):
        self.target_attraction_weight = target_attraction_weight
        self.obstacle_repulsion_weight = obstacle_repulsion_weight
        self.mesh_cohesion_weight = mesh_cohesion_weight
        self.comm_threshold_ratio = comm_threshold_ratio

    def select_actions(self, env: SwarmEnv) -> List[int]:
        """Compute discrete actions for all drones in the environment."""
        actions = []
        alive_drones = [d for d in env.drones if d.is_alive]
        active_targets = [t for t in env.targets if not t.is_reached]

        # Swarm centroid for cohesion
        if alive_drones:
            centroid = np.mean([d.position for d in alive_drones], axis=0)
        else:
            centroid = np.array([env.arena_width / 2.0, env.arena_height / 2.0])

        for drone in env.drones:
            if not drone.is_alive:
                actions.append(int(DroneAction.STAY))
                continue

            net_force = np.zeros(2, dtype=np.float32)

            # 1. Target Attraction
            if active_targets:
                # Assign closest target or distributed target
                dists = [
                    (np.linalg.norm(drone.position - t.position), t)
                    for t in active_targets
                ]
                _, target = min(dists, key=lambda x: x[0])
                to_target = target.position - drone.position
                dist_t = np.linalg.norm(to_target)
                if dist_t > 1e-4:
                    net_force += (to_target / dist_t) * self.target_attraction_weight

            # 2. Obstacle Repulsion
            for obs in env.obstacle_field.obstacles:
                to_obs = drone.position - obs.position
                dist_o = np.linalg.norm(to_obs) - obs.radius
                if 0 < dist_o < 3.5:
                    repulse_mag = (3.5 - dist_o) / 3.5
                    unit_repulse = to_obs / (np.linalg.norm(to_obs) + 1e-6)
                    net_force += unit_repulse * repulse_mag * self.obstacle_repulsion_weight

            # 3. Inter-drone Collision Repulsion (< 1.5)
            for other in alive_drones:
                if other.drone_id == drone.drone_id:
                    continue
                to_other = drone.position - other.position
                d_peer = np.linalg.norm(to_other)
                if 0 < d_peer < 1.5:
                    unit_peer = to_other / (d_peer + 1e-6)
                    net_force += unit_peer * ((1.5 - d_peer) / 1.5) * 3.0

            # 4. Mesh Cohesion Spring
            # If drone distance to centroid exceeds threshold, pull back towards mesh
            dist_to_center = np.linalg.norm(drone.position - centroid)
            max_allowed = env.comm_range * self.comm_threshold_ratio
            if dist_to_center > max_allowed:
                to_center = centroid - drone.position
                cohesion_mag = (dist_to_center - max_allowed) / max_allowed
                net_force += (to_center / (dist_to_center + 1e-6)) * cohesion_mag * self.mesh_cohesion_weight

            # 5. Boundary Repulsion
            margin = 2.0
            if drone.position[0] < margin:
                net_force[0] += (margin - drone.position[0]) * 2.0
            elif drone.position[0] > (env.arena_width - margin):
                net_force[0] -= (drone.position[0] - (env.arena_width - margin)) * 2.0

            if drone.position[1] < margin:
                net_force[1] += (margin - drone.position[1]) * 2.0
            elif drone.position[1] > (env.arena_height - margin):
                net_force[1] -= (drone.position[1] - (env.arena_height - margin)) * 2.0

            # Convert continuous net force into best matching discrete action
            if np.linalg.norm(net_force) < 0.2:
                best_action = DroneAction.STAY
            else:
                best_action = DroneAction.STAY
                best_alignment = -float("inf")
                for act in [DroneAction.NORTH, DroneAction.SOUTH, DroneAction.EAST, DroneAction.WEST]:
                    delta = ACTION_DELTAS[act]
                    alignment = np.dot(delta, net_force)
                    if alignment > best_alignment:
                        best_alignment = alignment
                        best_action = act

            actions.append(int(best_action))

        return actions


def run_heuristic_demo(
    scenario: str = "rescue_maze",
    seed: int = 42,
    max_steps: int = 60,
    verbose: bool = True,
) -> dict:
    """Execute a complete heuristic simulation episode and return summary metrics."""
    env = SwarmEnv(scenario=scenario, max_steps=max_steps)
    policy = HeuristicSwarmPolicy()

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    step = 0

    while step < max_steps:
        actions = policy.select_actions(env)
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        step += 1
        if terminated or truncated:
            break

    summary = {
        "scenario": scenario,
        "seed": seed,
        "steps": step,
        "total_reward": round(total_reward, 2),
        "coverage_ratio": round(info["coverage_ratio"], 3),
        "targets_reached": f"{info['targets_reached']}/{info['total_targets']}",
        "success": info["success"],
        "total_collisions": info["total_collisions"],
        "mesh_connected": info["mesh_connected"],
        "algebraic_connectivity": round(info["algebraic_connectivity"], 3),
    }

    if verbose:
        print("\n" + "=" * 50)
        print(f"Heuristic Simulation Demo [{scenario}]")
        print("=" * 50)
        for k, v in summary.items():
            print(f"  {k:<25}: {v}")
        print("=" * 50 + "\n")

    return {
        "summary": summary,
        "trajectory": env.trajectory_history,
    }
