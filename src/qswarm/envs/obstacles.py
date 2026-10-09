"""Static and dynamic obstacles in the emergency disaster environment."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    is_dynamic: bool = False
    vx: float = 0.0
    vy: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "radius": float(self.radius),
            "is_dynamic": self.is_dynamic,
            "vx": float(self.vx),
            "vy": float(self.vy),
        }


class ObstacleField:
    """Manages static debris and moving hazard vectors."""

    def __init__(
        self,
        arena_bounds: Tuple[float, float] = (20.0, 20.0),
        obstacles: List[Obstacle] = None,
    ):
        self.arena_width, self.arena_height = arena_bounds
        self.obstacles: List[Obstacle] = obstacles if obstacles is not None else []

    def step(self):
        """Update dynamic obstacles with boundary bouncing."""
        for obs in self.obstacles:
            if not obs.is_dynamic:
                continue

            obs.x += obs.vx
            obs.y += obs.vy

            # Bounding box collision and rebound
            if obs.x - obs.radius <= 0:
                obs.x = obs.radius
                obs.vx = abs(obs.vx)
            elif obs.x + obs.radius >= self.arena_width:
                obs.x = self.arena_width - obs.radius
                obs.vx = -abs(obs.vx)

            if obs.y - obs.radius <= 0:
                obs.y = obs.radius
                obs.vy = abs(obs.vy)
            elif obs.y + obs.radius >= self.arena_height:
                obs.y = self.arena_height - obs.radius
                obs.vy = -abs(obs.vy)

    def check_collision(self, point: np.ndarray, collision_padding: float = 0.2) -> bool:
        """Check if a point collides with any obstacle."""
        for obs in self.obstacles:
            dist = np.linalg.norm(point - obs.position)
            if dist < (obs.radius + collision_padding):
                return True
        return False

    def blocks_line_of_sight(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if line segment p1-p2 intersects any obstacle circle."""
        d = p2 - p1
        d_len = np.linalg.norm(d)
        if d_len < 1e-6:
            return False
        d_norm = d / d_len

        for obs in self.obstacles:
            center = obs.position
            # Projection of circle center onto segment
            t = np.dot(center - p1, d_norm)
            t_clamped = np.clip(t, 0.0, d_len)
            closest = p1 + t_clamped * d_norm
            dist = np.linalg.norm(closest - center)
            if dist < obs.radius:
                return True
        return False

    def get_radar_readings(
        self,
        position: np.ndarray,
        sensing_radius: float = 5.0,
        num_sectors: int = 4,
    ) -> np.ndarray:
        """Sector-based obstacle radar distances (North, South, East, West sectors)."""
        distances = np.full(num_sectors, sensing_radius, dtype=np.float32)
        # Sectors: 0: North (+y), 1: South (-y), 2: East (+x), 3: West (-x)
        for obs in self.obstacles:
            vec = obs.position - position
            dist = float(np.linalg.norm(vec)) - obs.radius
            if dist < 0:
                dist = 0.0
            if dist < sensing_radius:
                dx, dy = vec[0], vec[1]
                # Determine primary sector
                if abs(dy) >= abs(dx):
                    sector = 0 if dy > 0 else 1
                else:
                    sector = 2 if dx > 0 else 3
                distances[sector] = min(distances[sector], dist)

        # Normalize radar to [0, 1] range: 0 = collision contact, 1 = clear
        return distances / sensing_radius
