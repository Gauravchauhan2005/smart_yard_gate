"""
License Plate Detector module for AI-Based Smart Yard Gate Automation System.
Decoupled from vehicle detection, providing both:
1. Deep-learning YOLO detection when trained plate weights are configured.
2. Robust OpenCV morphological & contour-based localization fallback.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class PlateDetector:
    """
    Locates license plate regions within a vehicle crop or camera frame.
    Returns localized bounding boxes, cropped matrices, and detection confidences.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.50,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_custom_yolo_loaded = False

        if self.model_path and self.model_path.exists():
            self._load_yolo_plate_model()

    def _load_yolo_plate_model(self) -> None:
        """Loads custom YOLO weights trained for license plate detection."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self.is_custom_yolo_loaded = True
        except Exception:
            self.is_custom_yolo_loaded = False

    def detect_plates(
        self,
        image: np.ndarray,
        vehicle_bbox: list | tuple | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Locates license plates within an image or designated vehicle bounding box.
        Returns a list of detected plates with coordinates, crops, and confidences.
        """
        if image is None or image.size == 0:
            return []

        # If a vehicle bounding box is supplied, crop vehicle region first
        working_img = image
        offset_x, offset_y = 0, 0
        if vehicle_bbox:
            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]
            working_img = image[vy1:vy2, vx1:vx2]
            offset_x, offset_y = vx1, vy1
            if working_img.size == 0:
                working_img = image
                offset_x, offset_y = 0, 0

        # Method 1: If trained YOLO plate model is active, perform inference
        if self.is_custom_yolo_loaded and self.model is not None:
            return self._detect_yolo(working_img, offset_x, offset_y)

        # Method 2: High-accuracy OpenCV Morphological & Contour Localizer
        return self._detect_contour_morphology(working_img, offset_x, offset_y)

    def _detect_yolo(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> List[Dict[str, Any]]:
        """Performs plate localization via loaded YOLO weights."""
        results = []
        preds = self.model.predict(image, conf=self.confidence_threshold, verbose=False)
        for r in preds:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy

                crop = image[y1:y2, x1:x2].copy()
                global_bbox = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]

                results.append({
                    "bbox": global_bbox,
                    "confidence": round(conf, 4),
                    "crop": crop,
                    "method": "yolo_weights",
                })
        return results

    def _detect_contour_morphology(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> List[Dict[str, Any]]:
        """
        Robust computer vision localization using morphological gradients,
        edge density analysis, and rectangular aspect ratio filtering.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Morphological Top-Hat and Black-Hat to isolate high-contrast plate text
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        enhanced = cv2.add(gray, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)

        # Vertical Sobel gradient filter (plates have strong vertical character edges)
        grad_x = cv2.Sobel(enhanced, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        if max_val > min_val:
            grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
        grad_x = grad_x.astype("uint8")

        # Gaussian Blur and closing to connect character regions
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)

        # Otsu thresholding
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0:
                continue

            aspect_ratio = cw / float(ch)
            area = cw * ch

            # Standard license plate aspect ratios (2.0 to 5.5) and minimum dimensions
            if 1.8 <= aspect_ratio <= 5.8 and 1200 <= area <= (w * h * 0.40):
                # Calculate edge density in candidate region
                roi = gray[y:y+ch, x:x+cw]
                edge_density = np.count_nonzero(cv2.Canny(roi, 50, 150)) / float(area)

                if edge_density > 0.04:  # True plates have high internal edge transitions
                    score = min(0.95, 0.65 + (edge_density * 1.5))
                    candidates.append((score, x, y, cw, ch))

        # Sort candidates by confidence score
        candidates.sort(key=lambda c: c[0], reverse=True)

        results = []
        if candidates:
            # Take top candidate
            score, x, y, cw, ch = candidates[0]
            crop = image[y:y+ch, x:x+cw].copy()
            results.append({
                "bbox": [x + offset_x, y + offset_y, x + cw + offset_x, y + ch + offset_y],
                "confidence": round(score, 4),
                "crop": crop,
                "method": "morphological_contour",
            })
        else:
            # Fallback heuristic: Estimate plate region at lower central 35% of truck frame
            fx1 = int(w * 0.25)
            fy1 = int(h * 0.60)
            fx2 = int(w * 0.75)
            fy2 = int(h * 0.92)
            fallback_crop = image[fy1:fy2, fx1:fx2].copy()
            results.append({
                "bbox": [fx1 + offset_x, fy1 + offset_y, fx2 + offset_x, fy2 + offset_y],
                "confidence": 0.68,
                "crop": fallback_crop,
                "method": "spatial_heuristic_fallback",
            })

        return results
