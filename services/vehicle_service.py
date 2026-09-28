"""
Vehicle Service module for AI-Based Smart Yard Gate Automation System.
Orchestrates vehicle check-in, check-out, slot allocation, and yard telemetry.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from werkzeug.datastructures import FileStorage

from database import db, Vehicle, Gate, YardLocation, Detection
from services.upload_service import UploadService
from services.detection_service import DetectionService
from services.ocr_service import OCRService


class VehicleService:
    """
    Core business logic service handling logistics vehicle lifecycles,
    gate automated check-in, checkout, and parking bay allocation.
    """

    def __init__(self):
        self.upload_service = UploadService()
        self.detection_service = DetectionService()
        self.ocr_service = OCRService()

    def process_automated_check_in(
        self,
        file: FileStorage,
        gate_number: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end automated check-in workflow:
        1. Ingest & Validate Image
        2. Detect Truck (YOLO)
        3. Localize License Plate
        4. Preprocess & Extract Plate Text (PaddleOCR)
        5. Allocate Yard Location
        6. Persist Vehicle & Detection records in Database
        7. Set Status to 'Inside Yard' or 'Manual Review'
        """
        # Step 1: Upload & Validate Frame
        upload_meta = self.upload_service.save_file(file, subfolder=f"gate_{gate_number}")
        image_path = upload_meta["absolute_path"]

        # Step 2 & 3: Run YOLO Vehicle & Plate Detection
        detection_res = self.detection_service.process_frame(
            image_path, gate_number=gate_number, save_visuals=True
        )

        vehicle_info = detection_res["vehicle"]
        plate_crop = detection_res.get("plate_crop_matrix")

        # Step 4: Run OCR Pipeline on Plate Crop
        ocr_res = self.ocr_service.recognize_plate(plate_crop)

        license_plate = ocr_res["license_plate"]
        trailer_number = ocr_res["trailer_number"]
        ocr_conf = ocr_res["ocr_confidence"]
        detect_conf = vehicle_info["confidence"]

        # Determine Entry Status
        if ocr_res["needs_manual_review"] or detect_conf < 0.65:
            status = "Manual Review"
        else:
            status = "Inside Yard"

        # Step 5: Allocate Available Yard Location
        allocated_location = None
        available_bay = (
            YardLocation.query.filter_by(status="Available")
            .order_by(YardLocation.location_code)
            .first()
        )

        now = datetime.now(timezone.utc)

        # Step 6: Create Vehicle Record
        vehicle = Vehicle(
            license_plate=license_plate,
            trailer_number=trailer_number,
            vehicle_type=vehicle_info["type"],
            detection_confidence=detect_conf,
            ocr_confidence=ocr_conf,
            gate_number=gate_number,
            status=status,
            entry_time=now,
            exit_time=None,
        )
        db.session.add(vehicle)
        db.session.flush()

        # Link yard location
        if available_bay:
            available_bay.vehicle_id = vehicle.id
            available_bay.status = "Occupied" if status == "Inside Yard" else "Maintenance"
            allocated_location = available_bay.location_code

        # Step 7: Record Detection Events
        detection_entry = Detection(
            vehicle_id=vehicle.id,
            image_path=detection_res["visuals"]["annotated_image"] or upload_meta["file_path"],
            object_type=vehicle.vehicle_type,
            confidence=detect_conf,
            detected_text=license_plate,
            created_at=now,
        )
        db.session.add(detection_entry)
        db.session.commit()

        return {
            "success": True,
            "vehicle": vehicle.to_dict(),
            "allocated_location": allocated_location,
            "visuals": detection_res["visuals"],
            "detection_telemetry": {
                "vehicle_type": vehicle.vehicle_type,
                "detection_confidence": detect_conf,
                "ocr_confidence": ocr_conf,
                "engine": ocr_res.get("engine"),
                "status": status,
            },
        }

    def process_gate_check_out(
        self,
        identifier: Union[int, str],
        gate_number: int = 3,
    ) -> Dict[str, Any]:
        """
        Executes vehicle checkout workflow:
        1. Finds vehicle by ID or License Plate
        2. Verifies vehicle is currently inside yard
        3. Updates status to 'Checked Out' and records exit_time
        4. Releases allocated yard slot back to 'Available'
        """
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            vehicle = db.session.get(Vehicle, int(identifier))
        else:
            vehicle = Vehicle.query.filter_by(license_plate=str(identifier)).first()

        if not vehicle:
            raise ValueError(f"Vehicle identified by '{identifier}' was not found in yard registry.")

        if vehicle.status == "Checked Out":
            raise ValueError(f"Vehicle '{vehicle.license_plate}' is already recorded as Checked Out.")

        now = datetime.now(timezone.utc)
        vehicle.status = "Checked Out"
        vehicle.exit_time = now

        # Release allocated parking bay
        released_slot = None
        if vehicle.yard_location:
            released_slot = vehicle.yard_location.location_code
            vehicle.yard_location.status = "Available"
            vehicle.yard_location.vehicle_id = None

        db.session.commit()

        return {
            "success": True,
            "message": f"Vehicle '{vehicle.license_plate}' checked out successfully at Gate #{gate_number}.",
            "vehicle": vehicle.to_dict(),
            "released_location": released_slot,
            "exit_time": now.isoformat(),
        }
