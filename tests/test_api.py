"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from qswarm.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "heuristic" in data["available_policies"]
    assert "hybrid_vqc" in data["available_policies"]


def test_api_circuit_endpoint():
    response = client.get("/circuit")
    assert response.status_code == 200
    data = response.json()
    assert data["num_qubits"] == 4
    assert data["circuit_depth"] > 0
    assert "q_0" in data["circuit_diagram"] or "q" in data["circuit_diagram"]


def test_api_simulate_endpoint():
    req_payload = {
        "policy": "heuristic",
        "scenario": "open_field",
        "seed": 42,
        "max_steps": 10,
        "num_drones": 4,
    }
    response = client.post("/simulate", json=req_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["steps"] > 0
    assert data["frames_count"] > 0
    assert "total_reward" in data
