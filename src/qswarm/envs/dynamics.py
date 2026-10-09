"""Drone kinematic dynamics and battery models."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np


class DroneAction(IntEnum):
    STAY = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


ACTION_DELTAS = {
    DroneAction.STAY: np.array([0.0, 0.0], dtype=np.float32),
    DroneAction.NORTH: np.array([0.0, 1.0], dtype=np.float32),
    DroneAction.SOUTH: np.array([0.0, -1.0], dtype=np.float32),
    DroneAction.EAST: np.array([1.0, 0.0], dtype=np.float32),
    DroneAction.WEST: np.array([-1.0, 0.0], dtype=np.float32),
}


@dataclass
class DroneState:
    drone_id: int
    position: np.ndarray        # Real ground-truth position [x, y]
    noisy_position: np.ndarray  # GPS-denied perceived odometry [x, y]
    velocity: np.ndarray        # Recent velocity vector [vx, vy]
    battery: float              # Remaining battery percentage [0, 100]
    is_alive: bool = True
    in_collision: bool = False

    def copy(self) -> "DroneState":
        return DroneState(
            drone_id=self.drone_id,
            position=self.position.copy(),
            noisy_position=self.noisy_position.copy(),
            velocity=self.velocity.copy(),
            battery=float(self.battery),
            is_alive=self.is_alive,
            in_collision=self.in_collision,
        )


class SwarmDynamics:
    """Handles kinematic integration and battery depletion for the drone swarm."""

    def __init__(
        self,
        step_size: float = 1.0,
        battery_drain_move: float = 1.0,
        battery_drain_hover: float = 0.2,
        battery_drain_radio: float = 0.1,
        arena_bounds: Tuple[float, float] = (20.0, 20.0),
    ):
        self.step_size = step_size
        self.battery_drain_move = battery_drain_move
        self.battery_drain_hover = battery_drain_hover
        self.battery_drain_radio = battery_drain_radio
        self.arena_width, self.arena_height = arena_bounds

    def step_drone(
        self,
        drone: DroneState,
        action: int,
        gps_noise_std: float = 0.0,
        rng: np.random.Generator = None,
    ) -> DroneState:
        """Advance single drone state by action, updating position and battery."""
        if not drone.is_alive or drone.battery <= 0.0:
            drone.is_alive = False
            drone.velocity = np.zeros(2, dtype=np.float32)
            return drone

        action_enum = DroneAction(int(action))
        delta = ACTION_DELTAS[action_enum] * self.step_size
        new_pos = drone.position + delta

        # Clamp to arena bounds
        clamped_pos = np.clip(
            new_pos,
            [0.0, 0.0],
            [self.arena_width, self.arena_height],
        ).astype(np.float32)

        # Actual movement displacement
        actual_velocity = clamped_pos - drone.position
        drone.position = clamped_pos
        drone.velocity = actual_velocity

        # Battery consumption
        is_moving = np.linalg.norm(actual_velocity) > 1e-4
        drain = self.battery_drain_move if is_moving else self.battery_drain_hover
        drain += self.battery_drain_radio
        drone.battery = max(0.0, drone.battery - drain)

        if drone.battery <= 0.0:
            drone.is_alive = False

        # Simulated GPS-denied odometry estimation
        if rng is not None and gps_noise_std > 0:
            noise = rng.normal(0.0, gps_noise_std, size=2).astype(np.float32)
            drone.noisy_position = drone.position + noise
        else:
            drone.noisy_position = drone.position.copy()

        return drone
