"""Tests for SwarmEnv and dynamics."""

import numpy as np

from qswarm.envs.dynamics import DroneAction, DroneState, SwarmDynamics
from qswarm.envs.swarm_env import SwarmEnv


def test_drone_dynamics_step():
    dynamics = SwarmDynamics(step_size=1.0, arena_bounds=(10.0, 10.0))
    drone = DroneState(
        drone_id=0,
        position=np.array([5.0, 5.0], dtype=np.float32),
        noisy_position=np.array([5.0, 5.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        battery=100.0,
    )

    # Step NORTH (+y)
    drone = dynamics.step_drone(drone, DroneAction.NORTH)
    assert np.allclose(drone.position, [5.0, 6.0])
    assert drone.battery < 100.0
    assert drone.is_alive

    # Step WEST (-x)
    drone = dynamics.step_drone(drone, DroneAction.WEST)
    assert np.allclose(drone.position, [4.0, 6.0])


def test_drone_boundary_clamping():
    dynamics = SwarmDynamics(step_size=2.0, arena_bounds=(10.0, 10.0))
    drone = DroneState(
        drone_id=0,
        position=np.array([9.5, 9.5], dtype=np.float32),
        noisy_position=np.array([9.5, 9.5], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        battery=100.0,
    )

    # Move NORTH past border
    drone = dynamics.step_drone(drone, DroneAction.NORTH)
    assert drone.position[1] == 10.0  # Clamped


def test_swarm_env_reset_determinism():
    env = SwarmEnv(scenario="rescue_maze", num_drones=4)
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)

    assert np.allclose(obs1, obs2)
    assert info1["targets_reached"] == info2["targets_reached"]
    assert len(env.drones) == 4


def test_swarm_env_step_execution():
    env = SwarmEnv(scenario="rescue_maze", num_drones=4)
    obs, info = env.reset(seed=42)

    actions = [1, 2, 3, 4]  # North, South, East, West
    next_obs, reward, terminated, truncated, info = env.step(actions)

    assert next_obs.shape == (4, 12)
    assert isinstance(reward, float)
    assert "reward_breakdown" in info
    assert env.current_step == 1
