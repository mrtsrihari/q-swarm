"""Tests for feature compression and quantum encodings."""

import numpy as np
import torch

from qswarm.policies.features import (
    StateCompressor,
    compress_features_deterministic,
    encode_graph_aware,
    encode_simple_angle,
    extract_drone_features,
    get_encoding_feature_breakdown,
)


def test_extract_drone_features_shape_and_bounds():
    raw_obs = np.random.uniform(0.0, 1.0, size=12).astype(np.float32)
    feat = extract_drone_features(raw_obs)
    assert feat.shape == (12,)
    assert np.all(feat >= -1.0) and np.all(feat <= 1.0)


def test_compress_features_deterministic_bounds():
    raw_obs = np.random.uniform(0.0, 1.0, size=12).astype(np.float32)
    comp = compress_features_deterministic(raw_obs, target_dim=4)
    assert comp.shape == (4,)
    assert np.all(comp >= -np.pi) and np.all(comp <= np.pi)


def test_graph_aware_and_simple_encodings():
    raw_obs = np.random.uniform(0.0, 1.0, size=12).astype(np.float32)
    simple = encode_simple_angle(raw_obs, target_dim=4)
    graph = encode_graph_aware(raw_obs, target_dim=4)

    assert simple.shape == (4,)
    assert graph.shape == (4,)
    assert np.all(simple >= -np.pi) and np.all(simple <= np.pi)
    assert np.all(graph >= -np.pi) and np.all(graph <= np.pi)

    breakdown = get_encoding_feature_breakdown(raw_obs)
    assert "simple_angle" in breakdown
    assert "graph_aware" in breakdown


def test_state_compressor_module():
    compressor = StateCompressor(input_dim=12, compressed_dim=4)
    x = torch.randn(8, 12)
    out = compressor(x)
    assert out.shape == (8, 4)
    assert torch.all(out >= -np.pi) and torch.all(out <= np.pi)
