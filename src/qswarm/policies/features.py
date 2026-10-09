"""Observation feature pipeline, normalization, and quantum state encodings."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class StateCompressor(nn.Module):
    """Compresses 12-dimensional drone observation into 4 bounded quantum/compact features."""

    def __init__(self, input_dim: int = 12, compressed_dim: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        # Lightweight linear reduction
        self.proj = nn.Linear(input_dim, compressed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bounded into [-pi, pi] for angle encoding via tanh
        return torch.tanh(self.proj(x)) * np.pi


def extract_drone_features(raw_obs: np.ndarray) -> np.ndarray:
    """Preprocess and normalize raw 12-dim observation into fixed bounded range [-1, 1].

    Raw observation layout per drone (12 features):
    0: x_norm [-1, 1]
    1: y_norm [-1, 1]
    2: vx [-1, 1]
    3: vy [-1, 1]
    4: battery_norm [0, 1]
    5: radar_N [0, 1]
    6: radar_S [0, 1]
    7: radar_E [0, 1]
    8: radar_W [0, 1]
    9: target_dx [-1, 1]
    10: target_dy [-1, 1]
    11: mesh_deg_norm [0, 1]
    """
    feat = np.array(raw_obs, dtype=np.float32)
    # Re-center [0, 1] features to [-1, 1]
    feat[..., 4] = feat[..., 4] * 2.0 - 1.0  # battery
    feat[..., 5:9] = feat[..., 5:9] * 2.0 - 1.0  # radar
    feat[..., 11] = feat[..., 11] * 2.0 - 1.0  # mesh degree
    return np.clip(feat, -1.0, 1.0)


def encode_simple_angle(obs_drone: np.ndarray, target_dim: int = 4) -> np.ndarray:
    """Baseline non-graph angle encoding: pure kinematics and obstacle sensing.

    Ablation Baseline (No Graph Awareness):
    Feature 0: Arena position x [-pi, pi]
    Feature 1: Arena position y [-pi, pi]
    Feature 2: Velocity heading or speed [-pi, pi]
    Feature 3: Obstacle clearance [-pi, pi]
    """
    f = extract_drone_features(obs_drone)

    # 0. Spatial position x
    feat0 = f[0] * np.pi

    # 1. Spatial position y
    feat1 = f[1] * np.pi

    # 2. Velocity heading
    feat2 = np.arctan2(f[3], f[2] + 1e-6)

    # 3. Obstacle clearance
    min_radar = np.min(f[5:9])
    feat3 = min_radar * np.pi

    compressed = np.array([feat0, feat1, feat2, feat3], dtype=np.float32)
    return np.clip(compressed, -np.pi, np.pi)


def encode_graph_aware(obs_drone: np.ndarray, target_dim: int = 4) -> np.ndarray:
    """Graph-informed quantum state encoding synthesizing swarm topology & navigation.

    Feature 0: Relative target bearing & goal direction proxy: arctan2(dy, dx)
    Feature 1: Obstacle risk & directional clearance: min(radar)
    Feature 2: Swarm graph metrics: Local degree, link quality, and connectivity risk
    Feature 3: Energy reserve, kinetic speed, and mesh cohesion tension
    """
    f = extract_drone_features(obs_drone)

    # 0. Target navigation vector
    feat0 = np.arctan2(f[10], f[9])  # in [-pi, pi]

    # 1. Obstacle clearance (scaled so close proximity approaches -pi)
    min_radar = np.min(f[5:9])
    feat1 = min_radar * np.pi  # in [-pi, pi]

    # 2. Graph mesh connectivity & articulation risk
    # f[11] is normalized local node degree in [-1, 1]
    # Low degree / isolation produces negative angles; robust mesh produces positive angles
    feat2 = f[11] * np.pi  # in [-pi, pi]

    # 3. Energy reserve, speed, and cohesion tension
    speed = np.linalg.norm(f[2:4])
    feat3 = (f[4] * 0.7 - speed * 0.3) * np.pi  # in [-pi, pi]

    compressed = np.array([feat0, feat1, feat2, feat3], dtype=np.float32)
    return np.clip(compressed, -np.pi, np.pi)


def compress_features_deterministic(
    obs_drone: np.ndarray,
    target_dim: int = 4,
    encoding_type: str = "graph_aware",
) -> np.ndarray:
    """Configurable state compression dispatch."""
    if encoding_type == "simple_angle":
        return encode_simple_angle(obs_drone, target_dim=target_dim)
    return encode_graph_aware(obs_drone, target_dim=target_dim)


def get_encoding_feature_breakdown(obs_drone: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return explicit breakdown comparing both encodings for dashboard & analysis."""
    simple = encode_simple_angle(obs_drone, target_dim=4)
    graph = encode_graph_aware(obs_drone, target_dim=4)

    return {
        "simple_angle": {
            "x0 (Position X)": float(simple[0]),
            "x1 (Position Y)": float(simple[1]),
            "x2 (Heading Velocity)": float(simple[2]),
            "x3 (Clearance Radar)": float(simple[3]),
        },
        "graph_aware": {
            "x0 (Target Bearing)": float(graph[0]),
            "x1 (Obstacle Clearance)": float(graph[1]),
            "x2 (Swarm Mesh Degree)": float(graph[2]),
            "x3 (Energy & Cohesion)": float(graph[3]),
        },
    }
