"""GPS-denied localization sensing and search coverage models."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RescueTarget:
    target_id: int
    x: float
    y: float
    is_reached: bool = False
    reached_by_drone: int = -1

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "x": float(self.x),
            "y": float(self.y),
            "is_reached": self.is_reached,
            "reached_by_drone": self.reached_by_drone,
        }


class SearchCoverageGrid:
    """Discretized occupancy/exploration grid tracking search space coverage."""

    def __init__(
        self,
        arena_bounds: Tuple[float, float] = (20.0, 20.0),
        resolution: float = 1.0,
    ):
        self.arena_width, self.arena_height = arena_bounds
        self.resolution = resolution
        self.cols = int(np.ceil(self.arena_width / resolution))
        self.rows = int(np.ceil(self.arena_height / resolution))
        self.visited_grid = np.zeros((self.rows, self.cols), dtype=bool)

    def reset(self):
        self.visited_grid.fill(False)

    def update_coverage(self, positions: np.ndarray, sensing_radius: float = 2.0) -> int:
        """Mark cells within sensing radius of any active drone as visited. Returns newly visited cell count."""
        new_cells = 0
        r_cells = int(np.ceil(sensing_radius / self.resolution))

        for pos in positions:
            cx = int(pos[0] / self.resolution)
            cy = int(pos[1] / self.resolution)

            min_c = max(0, cx - r_cells)
            max_c = min(self.cols, cx + r_cells + 1)
            min_r = max(0, cy - r_cells)
            max_r = min(self.rows, cy + r_cells + 1)

            for r in range(min_r, max_r):
                for c in range(min_c, max_c):
                    cell_pos = np.array([(c + 0.5) * self.resolution, (r + 0.5) * self.resolution])
                    if np.linalg.norm(cell_pos - pos) <= sensing_radius:
                        if not self.visited_grid[r, c]:
                            self.visited_grid[r, c] = True
                            new_cells += 1

        return new_cells

    @property
    def coverage_ratio(self) -> float:
        total = self.rows * self.cols
        if total == 0:
            return 0.0
        return float(np.sum(self.visited_grid)) / float(total)


class GPSDeniedSensor:
    """Simulates realistic GPS denial with odometry drift and noise."""

    def __init__(self, noise_std: float = 0.15, drift_rate: float = 0.02):
        self.noise_std = noise_std
        self.drift_rate = drift_rate
        self.accumulated_drift = np.zeros(2, dtype=np.float32)

    def step_odometry(
        self,
        true_position: np.ndarray,
        velocity: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Estimate local position using noisy dead-reckoning."""
        drift_step = rng.normal(0.0, self.drift_rate, size=2).astype(np.float32)
        self.accumulated_drift += drift_step
        instant_noise = rng.normal(0.0, self.noise_std, size=2).astype(np.float32)
        return true_position + self.accumulated_drift + instant_noise
