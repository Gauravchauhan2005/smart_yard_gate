"""
OCR Service module for AI-Based Smart Yard Gate Automation System.
Coordinates optical character recognition, text normalization, error correction,
and confidence score verification.
"""

from typing import Dict, Any, Union, Optional
import cv2
import numpy as np
from detection.ocr import PlateOCR
from detection.preprocessing import ImagePreprocessor


class OCRService:
    """
    Coordinates character extraction from cropped license plate images.
    """

    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = confidence_threshold
        self.ocr_engine = PlateOCR(confidence_threshold=confidence_threshold)
        self.preprocessor = ImagePreprocessor()

    def recognize_plate(
        self,
        plate_image: np.ndarray,
        strategy: str = "standard",
    ) -> Dict[str, Any]:
        """
        Runs preprocessing and OCR on a cropped license plate matrix.
        Returns extracted text, confidence, and validation flags.
        """
        if plate_image is None or plate_image.size == 0:
            return {
                "license_plate": "UNREADABLE",
                "trailer_number": "—",
                "ocr_confidence": 0.0,
                "needs_manual_review": True,
                "error": "Empty plate crop",
            }

        # Ensure image has passed standard OpenCV preprocessing
        preprocessed, _ = self.preprocessor.preprocess_pipeline(plate_image, strategy=strategy)

        # Run OCR extraction
        ocr_result = self.ocr_engine.extract_text(preprocessed)

        plate_text = ocr_result.get("text", "")
        confidence = float(ocr_result.get("confidence", 0.0))

        # Check if confidence meets threshold
        needs_review = (
            not plate_text
            or plate_text == "UNREADABLE"
            or confidence < self.confidence_threshold
        )

        # Derive companion trailer tracking number
        clean_code = plate_text.replace("-", "")[-4:] if plate_text else "0000"
        trailer_number = f"TL-{clean_code}4-X"

        return {
            "license_plate": plate_text if plate_text else "UNKNOWN-PLATE",
            "raw_text": ocr_result.get("raw_text", ""),
            "trailer_number": trailer_number,
            "ocr_confidence": confidence,
            "engine": ocr_result.get("engine", "unknown"),
            "needs_manual_review": needs_review,
        }
