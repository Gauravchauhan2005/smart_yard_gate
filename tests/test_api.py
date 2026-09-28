"""
API test suite for AI-Based Smart Yard Gate Automation System.
Verifies health check endpoint, blueprint routing, and error handlers.
"""

import pytest
from app import create_app


@pytest.fixture
def client():
    """Create and configure a testing client for the application."""
    test_app = create_app("testing")
    with test_app.test_client() as test_client:
        yield test_client


def test_health_check_endpoint(client):
    """Test that the /api/health endpoint returns status 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data
    assert "timestamp" in data
    assert "phase" in data


def test_root_index_endpoint(client):
    """Test that the root URL returns service information."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.get_json()
    assert "endpoints" in data
    assert data["status"] == "ready"


def test_not_found_error_handler(client):
    """Test that 404 returns structured JSON error message."""
    response = client.get("/api/non-existent-endpoint")
    assert response.status_code == 404

    data = response.get_json()
    assert data["error"] == "Not Found"
    assert data["status_code"] == 404


def test_detect_placeholder_returns_501(client):
    """Test that unimplemented future endpoint returns 501."""
    response = client.post("/api/detect")
    assert response.status_code == 501
    data = response.get_json()
    assert data["error"] == "Not Implemented"
