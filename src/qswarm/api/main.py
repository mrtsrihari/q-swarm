"""FastAPI backend service for simulation, evaluation, and quantum telemetry."""

import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qswarm import __version__
from qswarm.envs.swarm_env import SwarmEnv
from qswarm.policies.classical import ClassicalMatchedPolicy, ClassicalMLPPolicy
from qswarm.policies.heuristic import HeuristicSwarmPolicy
from qswarm.policies.hybrid_vqc import HybridVQCPolicy

app = FastAPI(
    title="Q-Swarm RL API",
    description="Hybrid Quantum-Reinforcement Learning for Swarm Pathfinding & Dynamic Mesh Optimization",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulationRequest(BaseModel):
    policy: str = Field(default="heuristic", description="heuristic | classical | classical_matched | hybrid_vqc")
    scenario: str = Field(default="rescue_maze", description="open_field | rescue_maze | dynamic_hazard | mesh_tradeoff | scale_test")
    seed: int = Field(default=42, description="Random seed")
    max_steps: int = Field(default=40, description="Max episode steps")
    num_drones: int = Field(default=4, description="Drone swarm size")


class HealthResponse(BaseModel):
    status: str
    version: str
    available_policies: List[str]
    available_scenarios: List[str]


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        version=__version__,
        available_policies=["heuristic", "classical", "classical_matched", "hybrid_vqc"],
        available_scenarios=["open_field", "rescue_maze", "dynamic_hazard", "mesh_tradeoff", "scale_test"],
    )


@app.post("/simulate")
def run_simulation(req: SimulationRequest):
    """Run an episode simulation and return full trajectory frames."""
    env = SwarmEnv(
        scenario=req.scenario,
        num_drones=req.num_drones,
        max_steps=req.max_steps,
    )

    # Load requested policy
    if req.policy == "heuristic":
        policy = HeuristicSwarmPolicy()
    elif req.policy == "classical":
        ckpt = "results/checkpoints/classical/checkpoint.pt"
        policy = ClassicalMLPPolicy.load(ckpt) if os.path.exists(ckpt) else ClassicalMLPPolicy()
    elif req.policy == "classical_matched":
        ckpt = "results/checkpoints/classical_matched/checkpoint.pt"
        policy = ClassicalMatchedPolicy.load(ckpt) if os.path.exists(ckpt) else ClassicalMatchedPolicy()
    elif req.policy == "hybrid_vqc":
        ckpt = "results/checkpoints/hybrid_vqc/checkpoint.pt"
        policy = HybridVQCPolicy.load(ckpt) if os.path.exists(ckpt) else HybridVQCPolicy()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown policy: {req.policy}")

    obs, info = env.reset(seed=req.seed)
    total_reward = 0.0
    done = False
    step_count = 0

    while not done and step_count < req.max_steps:
        if req.policy == "heuristic":
            actions = policy.select_actions(env)
        else:
            actions = [policy.select_action(obs[i]) for i in range(env.num_drones)]

        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    # Format serializable trajectory frames
    frames = []
    for frame in env.trajectory_history:
        frames.append({
            "step": frame["step"],
            "drones": [
                {
                    "id": d.drone_id,
                    "pos": [round(float(d.position[0]), 3), round(float(d.position[1]), 3)],
                    "battery": round(float(d.battery), 1),
                    "alive": d.is_alive,
                }
                for d in frame["drones"]
            ],
            "edges": frame["edges"],
            "coverage_ratio": round(float(frame["coverage_ratio"]), 3),
            "reward": round(float(frame["reward"]), 2),
        })

    return {
        "scenario": req.scenario,
        "policy": req.policy,
        "seed": req.seed,
        "steps": step_count,
        "total_reward": round(total_reward, 2),
        "coverage_ratio": round(info.get("coverage_ratio", 0.0), 3),
        "success": info.get("success", False),
        "total_collisions": info.get("total_collisions", 0),
        "mesh_connected": info.get("mesh_connected", False),
        "frames_count": len(frames),
        "frames": frames,
    }


@app.get("/circuit")
def get_quantum_circuit():
    """Retrieve VQC schematic, depth, and parameter topology."""
    policy = HybridVQCPolicy(num_qubits=4, num_layers=2)
    return {
        "num_qubits": 4,
        "num_layers": 2,
        "circuit_depth": policy.get_circuit_depth(),
        "trainable_quantum_parameters": policy.count_quantum_parameters(),
        "total_hybrid_parameters": policy.count_parameters(),
        "circuit_diagram": policy.get_circuit_diagram(),
    }
