"""
Detection package for vehicle and license plate computer vision pipeline.
Exports YOLO VehicleDetector, PlateDetector, ImagePreprocessor, and PlateOCR modules.
"""

from .detector import VehicleDetector
from .plate_detector import PlateDetector
from .preprocessing import ImagePreprocessor
from .ocr import PlateOCR

__all__ = [
    "VehicleDetector",
    "PlateDetector",
    "ImagePreprocessor",
    "PlateOCR",
]
