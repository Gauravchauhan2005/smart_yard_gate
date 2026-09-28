"""
Unit test suite for Database models, relationships, and constraints.
Verifies Vehicle, Gate, YardLocation, and Detection ORM models using in-memory SQLite.
"""

from datetime import datetime, timezone
import pytest
from sqlalchemy.exc import IntegrityError
from app import create_app
from database import db, Vehicle, Gate, YardLocation, Detection


@pytest.fixture
def app():
    """Create and configure a testing Flask application context."""
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def session(app):
    """Provides an isolated database session per test."""
    return db.session


def test_gate_model_crud(session):
    """Test creating, querying, and serializing a Gate record."""
    gate = Gate(gate_number=1, gate_name="Gate #1 - JNPT Nhava Sheva Inbound", status="Online")
    session.add(gate)
    session.commit()

    saved_gate = Gate.query.filter_by(gate_number=1).first()
    assert saved_gate is not None
    assert saved_gate.gate_name == "Gate #1 - JNPT Nhava Sheva Inbound"
    assert saved_gate.status == "Online"

    d = saved_gate.to_dict()
    assert d["gate_number"] == 1
    assert d["status"] == "Online"


def test_duplicate_gate_number_constraint(session):
    """Test that unique constraint prevents duplicate gate numbers."""
    gate1 = Gate(gate_number=1, gate_name="Gate A", status="Online")
    session.add(gate1)
    session.commit()

    gate2 = Gate(gate_number=1, gate_name="Gate B", status="Offline")
    session.add(gate2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_vehicle_model_crud(session):
    """Test creating, querying, and serializing a Vehicle record."""
    vehicle = Vehicle(
        license_plate="MH-12-RN-8842",
        trailer_number="NL-01-T-8842",
        vehicle_type="Tata Prima 5530.S (Heavy Hauler)",
        detection_confidence=0.95,
        ocr_confidence=0.98,
        gate_number=1,
        status="Inside Yard",
    )
    session.add(vehicle)
    session.commit()

    saved_veh = Vehicle.query.filter_by(license_plate="MH-12-RN-8842").first()
    assert saved_veh is not None
    assert saved_veh.status == "Inside Yard"
    assert saved_veh.entry_time is not None
    assert saved_veh.created_at is not None

    d = saved_veh.to_dict()
    assert d["license_plate"] == "MH-12-RN-8842"
    assert d["detection_confidence"] == 0.95
    assert d["ocr_confidence"] == 0.98


def test_yard_location_relationship(session):
    """Test one-to-one relationship between Vehicle and YardLocation."""
    vehicle = Vehicle(
        license_plate="KA-01-MJ-5512",
        vehicle_type="Ashok Leyland 4220",
        gate_number=2,
        status="Inside Yard",
    )
    session.add(vehicle)
    session.commit()

    loc = YardLocation(location_code="Bay A-14", status="Occupied", vehicle_id=vehicle.id)
    session.add(loc)
    session.commit()

    # Query through vehicle relationship
    reloaded_vehicle = Vehicle.query.filter_by(license_plate="KA-01-MJ-5512").first()
    assert reloaded_vehicle.yard_location is not None
    assert reloaded_vehicle.yard_location.location_code == "Bay A-14"

    # Query through yard location
    reloaded_loc = YardLocation.query.filter_by(location_code="Bay A-14").first()
    assert reloaded_loc.vehicle is not None
    assert reloaded_loc.vehicle.license_plate == "KA-01-MJ-5512"


def test_detection_cascade_deletion(session):
    """Test that deleting a vehicle cascades to its child detections."""
    vehicle = Vehicle(license_plate="CA-7781-ZZ", status="Processing")
    session.add(vehicle)
    session.commit()

    det1 = Detection(
        vehicle_id=vehicle.id,
        object_type="truck",
        confidence=0.94,
        detected_text="CA-7781-ZZ",
    )
    det2 = Detection(
        vehicle_id=vehicle.id,
        object_type="license_plate",
        confidence=0.91,
        detected_text="CA-7781-ZZ",
    )
    session.add_all([det1, det2])
    session.commit()

    assert Detection.query.filter_by(vehicle_id=vehicle.id).count() == 2

    # Delete vehicle and assert detections are cascaded
    session.delete(vehicle)
    session.commit()

    assert Detection.query.filter_by(vehicle_id=vehicle.id).count() == 0


def test_vehicle_checkout_lifecycle(session):
    """Test vehicle lifecycle transitioning from Inside Yard to Checked Out."""
    vehicle = Vehicle(license_plate="DL-01-AB-1932", status="Inside Yard")
    session.add(vehicle)
    session.commit()

    # Simulate check-out
    now = datetime.now(timezone.utc)
    vehicle.status = "Checked Out"
    vehicle.exit_time = now
    session.commit()

    updated = session.get(Vehicle, vehicle.id)
    assert updated.status == "Checked Out"
    assert updated.exit_time is not None
