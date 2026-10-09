"""Tests for Classical and Parameter-Matched policy networks."""

import os

import numpy as np
import torch

from qswarm.policies.classical import ClassicalMatchedPolicy, ClassicalMLPPolicy


def test_classical_mlp_forward_and_action():
    model = ClassicalMLPPolicy(input_dim=12, hidden_dims=[32, 16], output_dim=5)
    x = torch.randn(4, 12)
    q_vals = model(x)
    assert q_vals.shape == (4, 5)

    obs = np.ones(12, dtype=np.float32) * 0.5
    action = model.select_action(obs, epsilon=0.0)
    assert 0 <= action < 5
    assert model.count_parameters() > 100


def test_classical_matched_policy_parameter_count():
    model = ClassicalMatchedPolicy(input_dim=4, output_dim=5)
    params = model.count_parameters()
    # 4 * 5 weights + 5 biases = 25 parameters
    assert params == 25

    obs = np.ones(12, dtype=np.float32) * 0.5
    action = model.select_action(obs, epsilon=0.0)
    assert 0 <= action < 5


def test_policy_save_load(tmp_path):
    model = ClassicalMLPPolicy(input_dim=12, hidden_dims=[32, 16], output_dim=5)
    save_file = os.path.join(tmp_path, "test_mlp.pt")
    model.save(save_file)

    loaded = ClassicalMLPPolicy.load(save_file)
    obs = np.ones(12, dtype=np.float32)
    assert model.select_action(obs) == loaded.select_action(obs)
