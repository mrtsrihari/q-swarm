"""Training pipeline for the Hybrid Variational Quantum Circuit (VQC) Policy."""

import argparse
import json
import os
import time
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from qswarm.envs.swarm_env import SwarmEnv
from qswarm.policies.features import compress_features_deterministic
from qswarm.policies.hybrid_vqc import HybridVQCPolicy
from qswarm.training.reproducibility import set_seed


class HybridReplayBuffer:
    def __init__(self, capacity: int = 2000):
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


def train_hybrid(
    config_path: str = "configs/model/hybrid_vqc.yaml",
    smoke: bool = False,
    seed: int = 42,
) -> Dict[str, any]:
    """Train the Hybrid VQC policy via end-to-end parameter-shift gradients."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    episodes = 4 if smoke else cfg["training"].get("episodes", 50)
    batch_size = 8 if smoke else cfg["model"].get("batch_size", 16)
    gamma = cfg["model"].get("gamma", 0.95)
    lr = cfg["model"].get("learning_rate", 0.01)
    eps_start = cfg["model"].get("epsilon_start", 1.0)
    eps_end = cfg["model"].get("epsilon_end", 0.05)
    eps_decay = 0.80 if smoke else cfg["model"].get("epsilon_decay", 0.95)
    target_update_interval = cfg["model"].get("target_update_interval", 5)

    num_qubits = cfg["model"].get("num_qubits", 4)
    num_layers = 1 if smoke else cfg["model"].get("num_layers", 2)
    entanglement = cfg["model"].get("entanglement", "ring")
    noise_model = cfg["model"].get("noise") if cfg["model"].get("backend") == "noisy_simulator" else None
    use_noise = (cfg["model"].get("backend") == "noisy_simulator")

    save_dir = cfg["training"].get("save_dir", "results/checkpoints/hybrid_vqc")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Hybrid Policy and Target
    policy = HybridVQCPolicy(
        num_qubits=num_qubits,
        num_layers=num_layers,
        output_dim=5,
        entanglement=entanglement,
        noise_model=noise_model,
        use_noise=use_noise,
    )
    target_policy = HybridVQCPolicy(
        num_qubits=num_qubits,
        num_layers=num_layers,
        output_dim=5,
        entanglement=entanglement,
    )
    target_policy.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    replay_buffer = HybridReplayBuffer(capacity=cfg["model"].get("buffer_size", 2000))

    env = SwarmEnv(scenario="rescue_maze", num_drones=4, max_steps=30 if smoke else 50)

    start_time = time.time()
    epsilon = eps_start
    episode_returns = []
    losses = []

    print("\n" + "=" * 55)
    print("Initiating Hybrid VQC Training...")
    print(f"  Qubits: {num_qubits} | Circuit Depth: {policy.get_circuit_depth()}")
    print(f"  Quantum Parameters: {policy.count_quantum_parameters()} | Total Parameters: {policy.count_parameters()}")
    print("=" * 55)

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            # Action selection
            actions = []
            for i in range(env.num_drones):
                act = policy.select_action(obs[i], epsilon=epsilon, rng=rng)
                actions.append(act)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            ep_reward += reward

            # Push transitions for each drone
            for i in range(env.num_drones):
                s = compress_features_deterministic(obs[i], target_dim=num_qubits)
                ns = compress_features_deterministic(next_obs[i], target_dim=num_qubits)
                replay_buffer.push(s, actions[i], reward / env.num_drones, ns, float(done))

            obs = next_obs

            # Optimize step
            if len(replay_buffer) >= batch_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                q_values = policy(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Fast target evaluation
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
        "model_type": "hybrid_vqc",
        "num_qubits": num_qubits,
        "circuit_depth": policy.get_circuit_depth(),
        "trainable_quantum_parameters": policy.count_quantum_parameters(),
        "trainable_parameters": policy.count_parameters(),
        "episodes": episodes,
        "wall_clock_time": round(train_time, 2),
        "mean_return": round(float(np.mean(episode_returns)), 2),
        "mean_loss": round(float(np.mean(losses)) if losses else 0.0, 4),
        "checkpoint": checkpoint_path,
    }

    with open(metrics_path, "w") as f:
        json.dump(train_metadata, f, indent=2)

    print("\n" + "=" * 55)
    print("Hybrid VQC Training Complete")
    print(f"  Episodes Completed   : {episodes}")
    print(f"  Wall-Clock Time      : {train_metadata['wall_clock_time']}s")
    print(f"  Mean Return          : {train_metadata['mean_return']}")
    print(f"  Checkpoint Saved To  : {checkpoint_path}")
    print("=" * 55 + "\n")

    return train_metadata


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid VQC Policy")
    parser.add_argument("--config", default="configs/model/hybrid_vqc.yaml", help="Path to config")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke training on CPU")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    args = parser.parse_args()

    train_hybrid(config_path=args.config, smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()
