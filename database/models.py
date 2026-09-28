"""
SQLAlchemy ORM models for AI-Based Smart Yard Gate Automation System.
Implements the core domain entities:
- Vehicle: Tracked truck/trailer lifecycle, plate, status, and AI confidences.
- Gate: Ingress/egress access lane status and cameras.
- YardLocation: Parking/staging bay allocation and vehicle occupancy.
- Detection: Computer vision inference events, bounding boxes, crops, and OCR strings.
"""

from datetime import datetime, timezone
from database.db import db


class Vehicle(db.Model):
    """
    Vehicle entity representing commercial trucks and trailers passing through yard gates.
    Tracks check-in, check-out, OCR extracted plate, and operational status.
    """

    __tablename__ = "vehicles"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    license_plate = db.Column(db.String(32), nullable=False, index=True)
    trailer_number = db.Column(db.String(32), nullable=True, index=True)
    vehicle_type = db.Column(db.String(64), nullable=False, default="Semi-Truck")
    detection_confidence = db.Column(db.Float, nullable=True, default=0.0)
    ocr_confidence = db.Column(db.Float, nullable=True, default=0.0)
    gate_number = db.Column(db.Integer, nullable=False, default=1, index=True)
    status = db.Column(db.String(32), nullable=False, default="Inside Yard", index=True)
    entry_time = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    exit_time = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    detections = db.relationship(
        "Detection",
        backref="vehicle",
        cascade="all, delete-orphan",
        lazy="select",
    )
    yard_location = db.relationship(
        "YardLocation",
        backref="vehicle",
        uselist=False,
        lazy="select",
    )

    def to_dict(self) -> dict:
        """Serializes vehicle record to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "license_plate": self.license_plate,
            "trailer_number": self.trailer_number,
            "vehicle_type": self.vehicle_type,
            "detection_confidence": round(self.detection_confidence, 4) if self.detection_confidence is not None else 0.0,
            "ocr_confidence": round(self.ocr_confidence, 4) if self.ocr_confidence is not None else 0.0,
            "gate_number": self.gate_number,
            "status": self.status,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "yard_location": self.yard_location.location_code if self.yard_location else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return f"<Vehicle id={self.id} plate='{self.license_plate}' status='{self.status}'>"


class Gate(db.Model):
    """
    Gate entity representing physical gate lanes at the facility perimeter.
    """

    __tablename__ = "gates"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    gate_number = db.Column(db.Integer, unique=True, nullable=False, index=True)
    gate_name = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(32), nullable=False, default="Online")

    def to_dict(self) -> dict:
        """Serializes gate entity to dictionary."""
        return {
            "id": self.id,
            "gate_number": self.gate_number,
            "gate_name": self.gate_name,
            "status": self.status,
        }

    def __repr__(self) -> str:
        return f"<Gate number={self.gate_number} name='{self.gate_name}' status='{self.status}'>"


class YardLocation(db.Model):
    """
    YardLocation entity representing staging bays, docks, and parking slots.
    """

    __tablename__ = "yard_locations"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    location_code = db.Column(db.String(64), unique=True, nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="Available")  # Available, Occupied, Maintenance
    vehicle_id = db.Column(
        db.Integer,
        db.ForeignKey("vehicles.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    def to_dict(self) -> dict:
        """Serializes yard location entity to dictionary."""
        return {
            "id": self.id,
            "location_code": self.location_code,
            "status": self.status,
            "vehicle_id": self.vehicle_id,
        }

    def __repr__(self) -> str:
        return f"<YardLocation code='{self.location_code}' status='{self.status}' vehicle_id={self.vehicle_id}>"


class Detection(db.Model):
    """
    Detection entity capturing individual computer vision inference events.
    Stores bounding boxes, confidence, classified object types, and optical crops.
    """

    __tablename__ = "detections"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    vehicle_id = db.Column(
        db.Integer,
        db.ForeignKey("vehicles.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    image_path = db.Column(db.String(255), nullable=True)
    object_type = db.Column(db.String(64), nullable=False)  # 'truck', 'trailer', 'license_plate'
    confidence = db.Column(db.Float, nullable=False, default=0.0)
    detected_text = db.Column(db.String(128), nullable=True)
    created_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        """Serializes detection record to dictionary."""
        return {
            "id": self.id,
            "vehicle_id": self.vehicle_id,
            "image_path": self.image_path,
            "object_type": self.object_type,
            "confidence": round(self.confidence, 4),
            "detected_text": self.detected_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<Detection id={self.id} type='{self.object_type}' conf={self.confidence:.2f}>"
