"""Graph-aware communication mesh computing using NetworkX."""

from dataclasses import dataclass
from typing import Dict

import networkx as nx
import numpy as np


@dataclass
class MeshMetrics:
    num_nodes: int
    num_edges: int
    num_components: int
    largest_component_ratio: float
    average_degree: float
    algebraic_connectivity: float
    is_fully_connected: bool
    mean_link_quality: float


class SwarmMeshNetwork:
    """Constructs and analyzes the dynamic communication mesh of the swarm."""

    def __init__(self, comm_range: float = 8.0, line_of_sight_enabled: bool = True):
        self.comm_range = comm_range
        self.line_of_sight_enabled = line_of_sight_enabled

    def build_graph(
        self,
        drone_positions: np.ndarray,  # Shape (N, 2)
        alive_mask: np.ndarray,       # Shape (N,) bool
        obstacle_field = None,
    ) -> nx.Graph:
        """Construct weighted undirected communication graph."""
        n = len(drone_positions)
        graph = nx.Graph()

        # Add all nodes
        for i in range(n):
            graph.add_node(
                i,
                pos=drone_positions[i].tolist(),
                alive=bool(alive_mask[i]),
            )

        # Connect pairwise if alive, within range, and unobstructed
        for i in range(n):
            if not alive_mask[i]:
                continue
            for j in range(i + 1, n):
                if not alive_mask[j]:
                    continue

                p_i = drone_positions[i]
                p_j = drone_positions[j]
                dist = float(np.linalg.norm(p_i - p_j))

                if dist <= self.comm_range:
                    # Check line of sight if enabled
                    blocked = False
                    if self.line_of_sight_enabled and obstacle_field is not None:
                        blocked = obstacle_field.blocks_line_of_sight(p_i, p_j)

                    if not blocked:
                        # Normalized link quality: 1.0 (touching) down to 0.0 (edge of range)
                        link_quality = max(0.0, 1.0 - (dist / self.comm_range))
                        graph.add_edge(i, j, weight=link_quality, distance=dist)

        return graph

    def compute_metrics(self, graph: nx.Graph, alive_count: int) -> MeshMetrics:
        """Compute comprehensive graph invariants and algebraic connectivity."""
        active_nodes = [n for n, d in graph.nodes(data=True) if d.get("alive", True)]
        n_active = len(active_nodes)

        if n_active <= 1:
            return MeshMetrics(
                num_nodes=n_active,
                num_edges=0,
                num_components=1 if n_active == 1 else 0,
                largest_component_ratio=1.0 if n_active == 1 else 0.0,
                average_degree=0.0,
                algebraic_connectivity=0.0,
                is_fully_connected=True if n_active == 1 else False,
                mean_link_quality=1.0 if n_active == 1 else 0.0,
            )

        subgraph = graph.subgraph(active_nodes)
        components = list(nx.connected_components(subgraph))
        num_components = len(components)
        largest_comp_size = max(len(c) for c in components) if components else 0
        largest_component_ratio = largest_comp_size / n_active

        num_edges = subgraph.number_of_edges()
        degrees = [d for _, d in subgraph.degree()]
        average_degree = float(np.mean(degrees)) if degrees else 0.0

        weights = [d["weight"] for _, _, d in subgraph.edges(data=True) if "weight" in d]
        mean_link_quality = float(np.mean(weights)) if weights else 0.0

        is_fully_connected = (num_components == 1)

        # Algebraic connectivity (Fiedler eigenvalue lambda_2 of Laplacian)
        algebraic_connectivity = 0.0
        if is_fully_connected and n_active >= 2:
            try:
                L = nx.laplacian_matrix(subgraph, weight="weight").toarray().astype(float)
                eigenvals = np.sort(np.linalg.eigvalsh(L))
                if len(eigenvals) >= 2:
                    algebraic_connectivity = float(max(0.0, eigenvals[1]))
            except Exception:
                algebraic_connectivity = 0.0

        return MeshMetrics(
            num_nodes=n_active,
            num_edges=num_edges,
            num_components=num_components,
            largest_component_ratio=float(largest_component_ratio),
            average_degree=float(average_degree),
            algebraic_connectivity=float(algebraic_connectivity),
            is_fully_connected=is_fully_connected,
            mean_link_quality=float(mean_link_quality),
        )

    def extract_node_features(self, graph: nx.Graph, drone_idx: int) -> Dict[str, float]:
        """Extract per-drone localized graph connectivity metrics."""
        if not graph.has_node(drone_idx):
            return {"degree": 0.0, "mean_link_quality": 0.0, "connectivity_risk": 1.0}

        neighbors = list(graph.neighbors(drone_idx))
        deg = len(neighbors)
        if deg == 0:
            return {"degree": 0.0, "mean_link_quality": 0.0, "connectivity_risk": 1.0}

        qualities = [graph[drone_idx][nbr].get("weight", 0.0) for nbr in neighbors]
        mean_quality = float(np.mean(qualities))

        # Risk flag if degree is 1 (articulation risk) or lowest link is dangerously weak
        min_quality = float(min(qualities))
        risk = 1.0 if (deg <= 1 or min_quality < 0.2) else 0.0

        return {
            "degree": float(deg),
            "mean_link_quality": mean_quality,
            "connectivity_risk": risk,
        }
