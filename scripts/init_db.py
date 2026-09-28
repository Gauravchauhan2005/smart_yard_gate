"""
Database initialization and seeding script for AI-Based Smart Yard Gate Automation System.
Creates all database tables and seeds initial gates, yard locations, and fleet records.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import create_app
from database import db, Vehicle, Gate, YardLocation, Detection


def seed_gates():
    """Seed standard gate lanes if not present."""
    default_gates = [
        {"gate_number": 1, "gate_name": "Gate #1 - JNPT Nhava Sheva Inbound", "status": "Online"},
        {"gate_number": 2, "gate_name": "Gate #2 - Bhiwandi Logistics Corridor Inbound", "status": "Online"},
        {"gate_number": 3, "gate_name": "Gate #3 - Mumbai-Pune Expressway Outbound", "status": "Online"},
        {"gate_number": 4, "gate_name": "Gate #4 - CONCOR DFC Rail Freight Siding", "status": "Online"},
    ]

    count = 0
    for g_data in default_gates:
        exists = Gate.query.filter_by(gate_number=g_data["gate_number"]).first()
        if not exists:
            gate = Gate(
                gate_number=g_data["gate_number"],
                gate_name=g_data["gate_name"],
                status=g_data["status"],
            )
            db.session.add(gate)
            count += 1

    db.session.commit()
    print(f"[*] Seeded {count} gate lanes.")


def seed_yard_locations():
    """Seed staging bays and parking slots."""
    locations = []

    # Zone A: Dry Van Staging Bays (A-01 to A-20)
    for i in range(1, 21):
        locations.append(f"Bay A-{i:02d}")

    # Zone B: Cold Storage Bays (B-01 to B-20)
    for i in range(1, 21):
        locations.append(f"Bay B-{i:02d}")

    # Zone C: Flatbed / Bulk Bays (C-01 to C-15)
    for i in range(1, 16):
        locations.append(f"Bay C-{i:02d}")

    # Inspection & Holding Bays
    locations.extend(["Inspection Bay 1", "Inspection Bay 2", "Holding Area 1"])

    count = 0
    for code in locations:
        exists = YardLocation.query.filter_by(location_code=code).first()
        if not exists:
            loc = YardLocation(location_code=code, status="Available")
            db.session.add(loc)
            count += 1

    db.session.commit()
    print(f"[*] Seeded {count} yard locations.")


def seed_vehicles():
    """Seed initial sample vehicles and link to yard locations."""
    if Vehicle.query.first():
        print("[*] Vehicle records already present. Skipping vehicle seed.")
        return

    now = datetime.now(timezone.utc)

    sample_vehicles = [
        {
            "license_plate": "MH-12-RN-8842",
            "trailer_number": "NL-01-T-8842",
            "vehicle_type": "Tata Prima 5530.S (Heavy Hauler)",
            "gate_number": 1,
            "status": "Inside Yard",
            "detection_confidence": 0.952,
            "ocr_confidence": 0.981,
            "entry_time": now - timedelta(hours=3, minutes=15),
            "exit_time": None,
            "location_code": "Bay A-14",
        },
        {
            "license_plate": "KA-01-MJ-5512",
            "trailer_number": "NL-01-T-5512",
            "vehicle_type": "Ashok Leyland 4220 (Multi-Axle)",
            "gate_number": 2,
            "status": "Inside Yard",
            "detection_confidence": 0.924,
            "ocr_confidence": 0.963,
            "entry_time": now - timedelta(hours=2, minutes=45),
            "exit_time": None,
            "location_code": "Bay B-04",
        },
        {
            "license_plate": "GJ-06-AX-3021",
            "trailer_number": "IND-TR-3021",
            "vehicle_type": "BharatBenz 3528C (Container)",
            "gate_number": 1,
            "status": "Checked Out",
            "detection_confidence": 0.961,
            "ocr_confidence": 0.978,
            "entry_time": now - timedelta(hours=5),
            "exit_time": now - timedelta(hours=1, minutes=30),
            "location_code": None,
        },
        {
            "license_plate": "DL-01-AB-1932",
            "trailer_number": "NL-01-T-1932",
            "vehicle_type": "Tata Signa 4825.TK (Tipper/Hauler)",
            "gate_number": 2,
            "status": "Manual Review",
            "detection_confidence": 0.742,
            "ocr_confidence": 0.612,
            "entry_time": now - timedelta(hours=1, minutes=15),
            "exit_time": None,
            "location_code": "Inspection Bay 2",
        },
        {
            "license_plate": "HR-26-DQ-7781",
            "trailer_number": "IND-TR-7781",
            "vehicle_type": "Eicher Pro 6048 (Flatbed)",
            "gate_number": 1,
            "status": "Inside Yard",
            "detection_confidence": 0.948,
            "ocr_confidence": 0.991,
            "entry_time": now - timedelta(minutes=48),
            "exit_time": None,
            "location_code": "Bay C-09",
        },
        {
            "license_plate": "TN-09-BX-6523",
            "trailer_number": "IND-TR-6523",
            "vehicle_type": "Tata LPT 3118 (Chemical Tanker)",
            "gate_number": 3,
            "status": "Processing",
            "detection_confidence": 0.912,
            "ocr_confidence": 0.940,
            "entry_time": now - timedelta(minutes=15),
            "exit_time": None,
            "location_code": "Holding Area 1",
        },
        {
            "license_plate": "WB-23-CD-9021",
            "trailer_number": "IND-TR-9021",
            "vehicle_type": "Ashok Leyland 5525 (Heavy Commercial)",
            "gate_number": 4,
            "status": "Inside Yard",
            "detection_confidence": 0.935,
            "ocr_confidence": 0.972,
            "entry_time": now - timedelta(minutes=30),
            "exit_time": None,
            "location_code": "Bay A-05",
        },
    ]

    for v_data in sample_vehicles:
        loc_code = v_data.pop("location_code")
        vehicle = Vehicle(**v_data)
        db.session.add(vehicle)
        db.session.flush()

        # Add corresponding detection record
        detection = Detection(
            vehicle_id=vehicle.id,
            image_path=f"static/uploads/capture_{vehicle.license_plate}.jpg",
            object_type=vehicle.vehicle_type,
            confidence=vehicle.detection_confidence,
            detected_text=vehicle.license_plate,
            created_at=vehicle.entry_time,
        )
        db.session.add(detection)

        # Allocate yard location if inside or manual review
        if loc_code:
            loc = YardLocation.query.filter_by(location_code=loc_code).first()
            if loc:
                loc.vehicle_id = vehicle.id
                loc.status = "Occupied" if vehicle.status == "Inside Yard" else "Maintenance"

    db.session.commit()
    print(f"[*] Seeded {len(sample_vehicles)} vehicles and detection records.")


def init_database(config_name: str = "development"):
    """Creates database schema and populates with seed records."""
    app = create_app(config_name)
    with app.app_context():
        print(f"[*] Target Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        print("[*] Creating database schema tables...")
        db.create_all()
        print("[+] Schema tables created successfully.")

        print("[*] Seeding default dataset...")
        seed_gates()
        seed_yard_locations()
        seed_vehicles()
        print("[+] Database initialization complete.")


if __name__ == "__main__":
    init_database()
