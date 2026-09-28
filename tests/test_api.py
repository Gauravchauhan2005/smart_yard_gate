"""
API and UI route test suite for AI-Based Smart Yard Gate Automation System.
Verifies health check endpoint, UI template rendering, and error handlers.
"""

import pytest
from app import create_app


from database import db, Vehicle, Gate, YardLocation


@pytest.fixture
def client():
    """Create and configure a testing client for the application."""
    test_app = create_app("testing")
    with test_app.app_context():
        db.create_all()
        # Seed test vehicle for detail view test
        v = Vehicle(
            id=101,
            license_plate="IL-8842-TR",
            trailer_number="TL-99014-X",
            vehicle_type="Semi-Truck",
            gate_number=1,
            status="Inside Yard",
            detection_confidence=0.95,
            ocr_confidence=0.98,
        )
        db.session.add(v)
        loc = YardLocation(location_code="Bay A-14", status="Occupied", vehicle_id=101)
        db.session.add(loc)
        db.session.commit()

    with test_app.test_client() as test_client:
        yield test_client

    with test_app.app_context():
        db.session.remove()
        db.drop_all()


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


def test_upload_missing_file_returns_400(client):
    """Test that POST /api/upload without file returns 400 Bad Request."""
    response = client.post("/api/upload")
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "Bad Request"


def test_upload_invalid_extension_returns_400(client):
    """Test that POST /api/upload with forbidden extension returns 400."""
    import io
    data = {
        "file": (io.BytesIO(b"malicious script"), "exploit.exe")
    }
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Validation Error"


def test_upload_corrupted_image_returns_400(client):
    """Test that POST /api/upload with corrupted image bytes returns 400."""
    import io
    data = {
        "file": (io.BytesIO(b"RANDOM_NON_IMAGE_BYTES_12345"), "truck.jpg")
    }
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["error"] == "Validation Error"


def test_upload_valid_image_returns_201(client):
    """Test that POST /api/upload with valid JPEG returns 201 Created."""
    import io
    from PIL import Image

    img = Image.new("RGB", (200, 150), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {
        "file": (buf, "test_truck.jpg"),
        "gate_number": "1",
    }
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 201
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["media_type"] == "image"
    assert json_data["data"]["width"] == 200
    assert json_data["data"]["height"] == 150


def test_detect_endpoint_with_valid_image(client):
    """Test that POST /api/detect validates and accepts incoming frame."""
    import io
    from PIL import Image

    img = Image.new("RGB", (320, 240), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {
        "file": (buf, "inbound_semi.jpg"),
        "gate_number": "2",
    }
    response = client.post("/api/detect", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["detection"]["ready"] is True
