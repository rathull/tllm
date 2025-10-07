# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path to import server modules
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from app import app

client = TestClient(app)

def test_health_endpoint():
    """Test that the health endpoint returns successfully"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"

def test_stats_endpoint_when_no_engine():
    """Test stats endpoint returns 503 when engine not initialized"""
    response = client.get("/stats")
    assert response.status_code == 503

def test_completion_endpoint_when_no_engine():
    """Test completion endpoint returns 503 when engine not initialized"""
    request_data = {
        "prompt": "Hello world",
        "max_tokens": 50
    }
    response = client.post("/v1/completions", json=request_data)
    assert response.status_code == 503

