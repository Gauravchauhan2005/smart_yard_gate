"""
Main web navigation routes blueprint.
Renders YMS dashboard, fleet vehicles, yard slot inventory, and analytics views.
"""

from datetime import datetime, timedelta, timezone
from flask import Blueprint, render_template, redirect, url_for, abort

main_bp = Blueprint("main", __name__)

# Sample mock dataset simulating active yard state (bridging until Phase 3 DB integration)
MOCK_KPIS = {
    "vehicles_today": 152,
    "currently_inside": 42,
    "automated_entries": 144,
    "avg_ocr_confidence": 0.968,
    "detection_accuracy": 0.945,
}

MOCK_GATES = [
    {"gate_number": 1, "gate_name": "Gate #1 - North Inbound", "status": "Online"},
    {"gate_number": 2, "gate_name": "Gate #2 - South Inbound", "status": "Online"},
    {"gate_number": 3, "gate_name": "Gate #3 - East Egress", "status": "Online"},
    {"gate_number": 4, "gate_name": "Gate #4 - West Rail Transfer", "status": "Online"},
]

MOCK_VEHICLES = [
    {
        "id": 101,
        "license_plate": "IL-8842-TR",
        "trailer_number": "TL-99014-X",
        "vehicle_type": "Semi-Truck",
        "gate_number": 1,
        "entry_time": "2026-09-28 08:14:22",
        "exit_time": None,
        "status": "Inside Yard",
        "detection_confidence": 0.95,
        "ocr_confidence": 0.98,
        "created_at": "2026-09-28 08:14:22",
    },
    {
        "id": 102,
        "license_plate": "TX-4019-BB",
        "trailer_number": "TL-55120-A",
        "vehicle_type": "Semi-Truck",
        "gate_number": 2,
        "entry_time": "2026-09-28 08:29:10",
        "exit_time": None,
        "status": "Inside Yard",
        "detection_confidence": 0.92,
        "ocr_confidence": 0.96,
        "created_at": "2026-09-28 08:29:10",
    },
    {
        "id": 103,
        "license_plate": "OH-1932-KL",
        "trailer_number": "TL-88231-M",
        "vehicle_type": "Box Truck",
        "gate_number": 1,
        "entry_time": "2026-09-28 08:45:01",
        "exit_time": "2026-09-28 10:30:15",
        "status": "Checked Out",
        "detection_confidence": 0.96,
        "ocr_confidence": 0.97,
        "created_at": "2026-09-28 08:45:01",
    },
    {
        "id": 104,
        "license_plate": "CA-7781-ZZ",
        "trailer_number": "TL-11004-D",
        "vehicle_type": "Semi-Truck",
        "gate_number": 2,
        "entry_time": "2026-09-28 09:12:44",
        "exit_time": None,
        "status": "Manual Review",
        "detection_confidence": 0.74,
        "ocr_confidence": 0.61,
        "created_at": "2026-09-28 09:12:44",
    },
    {
        "id": 105,
        "license_plate": "GA-3021-MM",
        "trailer_number": "TL-33921-R",
        "vehicle_type": "Flatbed",
        "gate_number": 1,
        "entry_time": "2026-09-28 09:35:18",
        "exit_time": None,
        "status": "Inside Yard",
        "detection_confidence": 0.94,
        "ocr_confidence": 0.99,
        "created_at": "2026-09-28 09:35:18",
    },
    {
        "id": 106,
        "license_plate": "PA-6523-XC",
        "trailer_number": "TL-77412-C",
        "vehicle_type": "Tanker",
        "gate_number": 3,
        "entry_time": "2026-09-28 09:50:55",
        "exit_time": None,
        "status": "Processing",
        "detection_confidence": 0.91,
        "ocr_confidence": 0.94,
        "created_at": "2026-09-28 09:50:55",
    },
    {
        "id": 107,
        "license_plate": "NY-9021-FK",
        "trailer_number": "TL-44102-E",
        "vehicle_type": "Semi-Truck",
        "gate_number": 1,
        "entry_time": "2026-09-28 10:02:11",
        "exit_time": "2026-09-28 11:45:00",
        "status": "Checked Out",
        "detection_confidence": 0.98,
        "ocr_confidence": 0.99,
        "created_at": "2026-09-28 10:02:11",
    }
]

MOCK_YARD = [
    {
        "vehicle_id": 101,
        "license_plate": "IL-8842-TR",
        "trailer_number": "TL-99014-X",
        "location_code": "Bay A-14",
        "vehicle_type": "Semi-Truck",
        "gate_number": 1,
        "entry_time": "2026-09-28 08:14:22",
        "status": "Inside Yard",
    },
    {
        "vehicle_id": 102,
        "license_plate": "TX-4019-BB",
        "trailer_number": "TL-55120-A",
        "location_code": "Bay B-04",
        "vehicle_type": "Semi-Truck",
        "gate_number": 2,
        "entry_time": "2026-09-28 08:29:10",
        "status": "Inside Yard",
    },
    {
        "vehicle_id": 104,
        "license_plate": "CA-7781-ZZ",
        "trailer_number": "TL-11004-D",
        "location_code": "Inspection Bay 2",
        "vehicle_type": "Semi-Truck",
        "gate_number": 2,
        "entry_time": "2026-09-28 09:12:44",
        "status": "Manual Review",
    },
    {
        "vehicle_id": 105,
        "license_plate": "GA-3021-MM",
        "trailer_number": "TL-33921-R",
        "location_code": "Bay C-09",
        "vehicle_type": "Flatbed",
        "gate_number": 1,
        "entry_time": "2026-09-28 09:35:18",
        "status": "Inside Yard",
    },
    {
        "vehicle_id": 106,
        "license_plate": "PA-6523-XC",
        "trailer_number": "TL-77412-C",
        "location_code": "Holding Area 1",
        "vehicle_type": "Tanker",
        "gate_number": 3,
        "entry_time": "2026-09-28 09:50:55",
        "status": "Processing",
    }
]

MOCK_ANALYTICS = {
    "total_vehicles": 152,
    "automated_entries": 144,
    "manual_reviews": 8,
    "avg_detection_confidence": 0.945,
    "avg_ocr_confidence": 0.968,
}


@main_bp.route("/")
def index():
    """Root route redirects directly to the operational dashboard."""
    return redirect(url_for("main.dashboard"))


@main_bp.route("/dashboard")
def dashboard():
    """Renders the main YMS dashboard."""
    recent_activity = [
        {
            "id": v["id"],
            "vehicle_type": v["vehicle_type"],
            "license_plate": v["license_plate"],
            "trailer_number": v["trailer_number"],
            "gate_number": v["gate_number"],
            "entry_time": v["entry_time"],
            "status": v["status"],
            "confidence": v["detection_confidence"],
        }
        for v in MOCK_VEHICLES
    ]
    return render_template(
        "index.html",
        active_page="dashboard",
        kpis=MOCK_KPIS,
        gates=MOCK_GATES,
        recent_activity=recent_activity,
    )


@main_bp.route("/vehicles")
def vehicles():
    """Renders the vehicle registry & history log."""
    return render_template(
        "vehicles.html",
        active_page="vehicles",
        vehicles=MOCK_VEHICLES,
    )


@main_bp.route("/vehicles/<int:vehicle_id>")
def vehicle_detail(vehicle_id: int):
    """Renders individual vehicle inspection and audit details."""
    vehicle = next((v for v in MOCK_VEHICLES if v["id"] == vehicle_id), None)
    if not vehicle:
        # Fallback profile for arbitrary ID in Phase 2
        vehicle = {
            "id": vehicle_id,
            "license_plate": f"ST-{vehicle_id:04d}-US",
            "trailer_number": f"TL-{vehicle_id * 11}-B",
            "vehicle_type": "Semi-Truck",
            "gate_number": 1,
            "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": None,
            "status": "Inside Yard",
            "detection_confidence": 0.94,
            "ocr_confidence": 0.96,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }

    return render_template(
        "vehicle_detail.html",
        active_page="vehicles",
        vehicle=vehicle,
    )


@main_bp.route("/yard")
def yard():
    """Renders yard slot occupancy and allocation inventory."""
    return render_template(
        "yard.html",
        active_page="yard",
        yard_items=MOCK_YARD,
    )


@main_bp.route("/analytics")
def analytics():
    """Renders analytics, KPIs, and visualization charts."""
    return render_template(
        "analytics.html",
        active_page="analytics",
        analytics_data=MOCK_ANALYTICS,
    )
