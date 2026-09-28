"""
API and UI route test suite for AI-Based Smart Yard Gate Automation System.
Verifies health check endpoint, UI template rendering, and error handlers.
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


def test_root_index_redirects_to_dashboard(client):
    """Test that root URL redirects to the operational dashboard."""
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert b"YMS Gate Operations Dashboard" in response.data


def test_dashboard_page_renders(client):
    """Test that the dashboard renders KPI cards and recent gate activity."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Vehicles Today" in response.data
    assert b"Currently Inside" in response.data
    assert b"Automated Entries" in response.data
    assert b"Avg OCR Confidence" in response.data
    assert b"Detection Accuracy" in response.data
    assert b"Recent Gate Activity" in response.data


def test_gate_automation_page_renders(client):
    """Test that the gate automation terminal renders with upload controls and vision output."""
    response = client.get("/gate/")
    assert response.status_code == 200
    assert b"Gate Ingress Terminal" in response.data
    assert b"Multi-Stage Inspection Results" in response.data
    assert b"1. ORIGINAL GATE CAPTURE" in response.data
    assert b"2. ANNOTATED YOLO DETECTIONS" in response.data


def test_vehicles_page_renders(client):
    """Test that the vehicle registry and search/filter interface renders."""
    response = client.get("/vehicles")
    assert response.status_code == 200
    assert b"Fleet & Vehicle Registry" in response.data
    assert b"Vehicle Movement Logs" in response.data


def test_vehicle_detail_page_renders(client):
    """Test that individual vehicle audit view renders properly."""
    response = client.get("/vehicles/101")
    assert response.status_code == 200
    assert b"Vehicle Inspection & Audit Record #101" in response.data
    assert b"IL-8842-TR" in response.data


def test_yard_inventory_page_renders(client):
    """Test that yard inventory and slot allocation page renders."""
    response = client.get("/yard")
    assert response.status_code == 200
    assert b"Yard Inventory & Slot Allocation" in response.data
    assert b"Current Yard Occupancy & Assigned Locations" in response.data


def test_analytics_page_renders(client):
    """Test that analytics view renders charts and metrics."""
    response = client.get("/analytics")
    assert response.status_code == 200
    assert b"Operational Intelligence & AI Analytics" in response.data
    assert b"Vehicles by Hour" in response.data


def test_not_found_error_handler(client):
    """Test that 404 returns structured JSON error message for API routes."""
    response = client.get("/api/non-existent-endpoint")
    assert response.status_code == 404
    data = response.get_json()
    assert data["error"] == "Not Found"
    assert data["status_code"] == 404


def test_detect_placeholder_returns_501(client):
    """Test that unimplemented future API endpoint returns 501."""
    response = client.post("/api/detect")
    assert response.status_code == 501
    data = response.get_json()
    assert data["error"] == "Not Implemented"
