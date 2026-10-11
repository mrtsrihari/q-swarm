"""DQN training loop for Classical MLP and Parameter-Matched baselines."""

import argparse
import json
import os
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from qswarm.envs.swarm_env import SwarmEnv
from qswarm.policies.classical import ClassicalMatchedPolicy, ClassicalMLPPolicy
from qswarm.policies.features import compress_features_deterministic, extract_drone_features
from qswarm.training.reproducibility import set_seed


class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def train_classical(
    config_path: str = "configs/model/classical.yaml",
    smoke: bool = False,
    model_type: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train a classical DQN agent with logging and checkpoint export."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    m_type = model_type or cfg["model"].get("type", "classical")
    is_matched = (m_type == "classical_matched")

    # Set parameters
    episodes = 5 if smoke else cfg["training"].get("episodes", 80)
    batch_size = 16 if smoke else cfg["model"].get("batch_size", 32)
    gamma = cfg["model"].get("gamma", 0.95)
    lr = cfg["model"].get("learning_rate", 0.001)
    eps_start = cfg["model"].get("epsilon_start", 1.0)
    eps_end = cfg["model"].get("epsilon_end", 0.05)
    eps_decay = 0.85 if smoke else cfg["model"].get("epsilon_decay", 0.95)
    target_update_interval = cfg["model"].get("target_update_interval", 5)

    save_dir = cfg["training"].get("save_dir", "results/checkpoints/classical")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize policy
    if is_matched:
        policy = ClassicalMatchedPolicy(input_dim=4, output_dim=5)
        target_policy = ClassicalMatchedPolicy(input_dim=4, output_dim=5)
    else:
        hidden_dims = cfg["model"].get("hidden_dims", [64, 32])
        policy = ClassicalMLPPolicy(input_dim=12, hidden_dims=hidden_dims, output_dim=5)
        target_policy = ClassicalMLPPolicy(input_dim=12, hidden_dims=hidden_dims, output_dim=5)

    target_policy.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(capacity=cfg["model"].get("buffer_size", 5000))

    env = SwarmEnv(scenario="rescue_maze", num_drones=4, max_steps=40 if smoke else 60)

    start_time = time.time()
    epsilon = eps_start
    episode_returns = []
    losses = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            # Multi-agent action selection
            actions = []
            for i in range(env.num_drones):
                act = policy.select_action(obs[i], epsilon=epsilon, rng=rng)
                actions.append(act)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            ep_reward += reward

            # Push transitions for each drone
            for i in range(env.num_drones):
                if is_matched:
                    s = compress_features_deterministic(obs[i], target_dim=4)
                    ns = compress_features_deterministic(next_obs[i], target_dim=4)
                else:
                    s = extract_drone_features(obs[i])
                    ns = extract_drone_features(next_obs[i])

                replay_buffer.push(s, actions[i], reward / env.num_drones, ns, float(done))

            obs = next_obs

            # Optimize step
            if len(replay_buffer) >= batch_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                q_values = policy(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    max_next_q = target_policy(b_ns).max(dim=1)[0]
                    target_q = b_r + (1.0 - b_d) * gamma * max_next_q

                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

        episode_returns.append(ep_reward)
        epsilon = max(eps_end, epsilon * eps_decay)

        if (ep + 1) % target_update_interval == 0:
            target_policy.load_state_dict(policy.state_dict())

    train_time = time.time() - start_time
    checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
    policy.save(checkpoint_path)

    metrics_path = os.path.join(save_dir, "train_metrics.json")
    train_metadata = {
        "model_type": m_type,
        "trainable_parameters": policy.count_parameters(),
        "episodes": episodes,
        "wall_clock_time": round(train_time, 2),
        "mean_return": round(float(np.mean(episode_returns)), 2),
        "last_return": round(float(episode_returns[-1]), 2),
        "mean_loss": round(float(np.mean(losses)) if losses else 0.0, 4),
        "checkpoint": checkpoint_path,
    }

    with open(metrics_path, "w") as f:
        json.dump(train_metadata, f, indent=2)

    print("\n" + "=" * 55)
    print(f"Classical Training Complete [{m_type}]")
    print(f"  Trainable Parameters : {train_metadata['trainable_parameters']}")
    print(f"  Episodes Completed   : {episodes}")
    print(f"  Wall-Clock Time      : {train_metadata['wall_clock_time']}s")
    print(f"  Final Mean Return    : {train_metadata['mean_return']}")
    print(f"  Checkpoint Saved To  : {checkpoint_path}")
    print("=" * 55 + "\n")

    return train_metadata


def main():
    parser = argparse.ArgumentParser(description="Train classical RL baseline")
    parser.add_argument("--config", default="configs/model/classical.yaml", help="Path to config")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke training on CPU")
    parser.add_argument("--matched", action="store_true", help="Train parameter-matched model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    model_type = "classical_matched" if args.matched else None
    train_classical(config_path=args.config, smoke=args.smoke, model_type=model_type, seed=args.seed)


if __name__ == "__main__":
    main()
