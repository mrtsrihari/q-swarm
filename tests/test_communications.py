"""Tests for communication mesh graph and algebraic connectivity."""

import numpy as np

from qswarm.envs.communications import SwarmMeshNetwork
from qswarm.envs.obstacles import Obstacle, ObstacleField


def test_mesh_fully_connected():
    mesh = SwarmMeshNetwork(comm_range=10.0, line_of_sight_enabled=False)
    positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 2.0],
        [0.0, 2.0],
    ], dtype=np.float32)
    alive_mask = np.array([True, True, True, True])

    graph = mesh.build_graph(positions, alive_mask)
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 6  # Complete graph K_4

    metrics = mesh.compute_metrics(graph, alive_count=4)
    assert metrics.is_fully_connected
    assert metrics.num_components == 1
    assert metrics.largest_component_ratio == 1.0
    assert metrics.algebraic_connectivity > 0.0


def test_mesh_disconnected_partition():
    mesh = SwarmMeshNetwork(comm_range=5.0, line_of_sight_enabled=False)
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [20.0, 20.0],
        [21.0, 20.0],
    ], dtype=np.float32)
    alive_mask = np.array([True, True, True, True])

    graph = mesh.build_graph(positions, alive_mask)
    metrics = mesh.compute_metrics(graph, alive_count=4)

    assert not metrics.is_fully_connected
    assert metrics.num_components == 2
    assert metrics.largest_component_ratio == 0.5
    assert metrics.algebraic_connectivity == 0.0


def test_mesh_line_of_sight_blockage():
    # Obstacle placed right between (0, 0) and (4, 0)
    obs = Obstacle(x=2.0, y=0.0, radius=1.0)
    obstacle_field = ObstacleField((10.0, 10.0), [obs])

    mesh = SwarmMeshNetwork(comm_range=10.0, line_of_sight_enabled=True)
    positions = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
    ], dtype=np.float32)
    alive_mask = np.array([True, True])

    graph = mesh.build_graph(positions, alive_mask, obstacle_field)
    assert graph.number_of_edges() == 0  # Blocked by obstacle
