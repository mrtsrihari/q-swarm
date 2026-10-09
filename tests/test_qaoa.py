"""Tests for QAOA Mesh Relay Selection and Exact/Greedy solvers."""

import networkx as nx

from qswarm.optimization.qaoa_mesh_repair import (
    ExactRelaySolver,
    GreedyRelaySolver,
    QAOARelayOptimizer,
    run_qaoa_benchmark,
)


def test_exact_and_greedy_solvers():
    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=0.8)
    g.add_edge(2, 3, weight=0.5)

    exact_val, exact_sol, exact_time = ExactRelaySolver.solve(g)
    assert exact_val > 0.0
    assert len(exact_sol) == 4
    assert exact_time < 0.5

    greedy_val, greedy_sol, greedy_time = GreedyRelaySolver.solve(g)
    assert greedy_val > 0.0
    assert len(greedy_sol) == 4
    assert greedy_time < 0.5


def test_qaoa_relay_optimizer_execution():
    g = nx.Graph()
    g.add_edge(0, 1, weight=0.9)
    g.add_edge(1, 2, weight=0.7)
    g.add_edge(2, 3, weight=0.4)

    qaoa = QAOARelayOptimizer(g, p_steps=1)
    val, sol, elapsed, depth, cx_count = qaoa.solve(max_iter=10)

    assert val > 0.0
    assert len(sol) == 4
    assert elapsed < 5.0
    assert depth > 0
    assert cx_count > 0


def test_run_qaoa_benchmark_end_to_end():
    res = run_qaoa_benchmark(num_nodes=4)
    assert res.qaoa_approx_ratio > 0.0
    assert res.exact_optimal_val >= res.qaoa_val
    assert res.graph_nodes == 4
