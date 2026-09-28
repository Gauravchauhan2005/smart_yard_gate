"""
OpenCV Image Preprocessing module for License Plate & Identifier Enhancement.
Provides a comprehensive suite of image transformations designed to maximize OCR accuracy:
- Bounding box cropping with safe bounds and padding margins
- Aspect-ratio preserving and standardized height resizing
- Grayscale conversion
- Edge-preserving noise reduction (Bilateral filter & Gaussian blur)
- Local contrast enhancement (CLAHE)
- Image sharpening via unsharp masking
- Adaptive and Otsu thresholding
- Rotation & skew perspective correction
"""

import math
from typing import Tuple, Dict, Any
import cv2
import numpy as np


class ImagePreprocessor:
    """Reusable OpenCV preprocessing algorithms for ANPR / YMS vision pipelines."""

    @staticmethod
    def crop_region(
        image: np.ndarray,
        bbox: list | tuple,
        margin: int = 4,
    ) -> np.ndarray:
        """
        Crops region of interest from an image matrix with padding margins.
        bbox format: [x1, y1, x2, y2]
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")

        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Ensure integers and apply margin padding
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1) - margin)
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2) + margin)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box coordinates after clipping: [{x1}, {y1}, {x2}, {y2}]")

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def resize_plate(
        image: np.ndarray,
        target_height: int = 64,
        target_width: int | None = None,
    ) -> np.ndarray:
        """
        Standardizes license plate dimensions while preserving aspect ratio.
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        if target_width is not None:
            return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        aspect_ratio = w / float(h)
        new_width = max(32, int(target_height * aspect_ratio))
        return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Converts BGR image to single-channel 8-bit grayscale."""
        if image is None:
            return None
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def reduce_noise(
        image: np.ndarray,
        method: str = "bilateral",
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ) -> np.ndarray:
        """
        Reduces optical noise and sensor grain.
        Bilateral filtering is preferred as it preserves sharp character edges.
        """
        if method == "bilateral":
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 3)
        return image

    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        clip_limit: float = 2.5,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Compensates for uneven lighting, headlight glare, and shadows at gate cameras.
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """
        Sharpens blurry plate characters using an unsharp masking convolution kernel.
        """
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def apply_threshold(
        image: np.ndarray,
        method: str = "otsu",
        block_size: int = 19,
        c: int = 9,
    ) -> np.ndarray:
        """
        Binarizes image matrix into high-contrast black/white characters.
        Methods:
        - 'otsu': Global optimal bimodal thresholding.
        - 'adaptive_gaussian': Local weighted Gaussian thresholding (resilient to uneven shadows).
        - 'adaptive_mean': Local arithmetic mean thresholding.
        """
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == "adaptive_gaussian":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
            )
        elif method == "adaptive_mean":
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
            )
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return binary

    @staticmethod
    def correct_perspective(image: np.ndarray) -> np.ndarray:
        """
        Detects plate orientation angle and deskews image horizontally.
        """
        if image is None or image.size == 0:
            return image

        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return image

        # Calculate median slope angle of detected lines
        angles = []
        for line in lines:
            coords = line.flatten()
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[:4]
                if x2 - x1 != 0:
                    angle = math.degrees(math.atan2(int(y2) - int(y1), int(x2) - int(x1)))
                    if abs(angle) < 45.0:  # Only consider slight tilt angles
                        angles.append(angle)

        if not angles:
            return image

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:  # Negligible skew
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed = cv2.warpAffine(
            image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed

    @classmethod
    def preprocess_pipeline(
        cls,
        plate_image: np.ndarray,
        strategy: str = "standard",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes an end-to-end preprocessing pipeline for OCR ingestion.
        Strategies:
        - 'standard': Crop -> Resize -> Grayscale -> Bilateral Denoise -> CLAHE -> Otsu -> Deskew
        - 'adaptive': Crop -> Resize -> Grayscale -> Gaussian Blur -> Adaptive Gaussian Threshold
        - 'contrast_boost': Crop -> Resize -> Grayscale -> Strong CLAHE -> Sharpen
        - 'grayscale_clean': Crop -> Resize -> Grayscale -> Bilateral Filter (unthresholded)
        """
        if plate_image is None or plate_image.size == 0:
            raise ValueError("Input plate image is empty or None.")

        # 1. Standardize size (target height 64px)
        resized = cls.resize_plate(plate_image, target_height=64)

        # 2. Grayscale conversion
        gray = cls.to_grayscale(resized)

        if strategy == "grayscale_clean":
            denoised = cls.reduce_noise(gray, method="bilateral")
            deskewed = cls.correct_perspective(denoised)
            return deskewed, {"strategy": strategy, "steps": ["resize", "grayscale", "bilateral", "deskew"]}

        elif strategy == "adaptive":
            blurred = cls.reduce_noise(gray, method="gaussian")
            thresholded = cls.apply_threshold(blurred, method="adaptive_gaussian")
            deskewed = cls.correct_perspective(thresholded)
            return deskewed, {"strategy": strategy, "steps": ["resize", "grayscale", "gaussian", "adaptive_thresh", "deskew"]}

        elif strategy == "contrast_boost":
            enhanced = cls.enhance_contrast(gray, clip_limit=3.0)
            sharpened = cls.sharpen(enhanced)
            thresholded = cls.apply_threshold(sharpened, method="otsu")
            deskewed = cls.correct_perspective(thresholded)
            return deskewed, {"strategy": strategy, "steps": ["resize", "grayscale", "clahe", "sharpen", "otsu", "deskew"]}

        else:  # 'standard'
            denoised = cls.reduce_noise(gray, method="bilateral")
            enhanced = cls.enhance_contrast(denoised, clip_limit=2.0)
            sharpened = cls.sharpen(enhanced)
            thresholded = cls.apply_threshold(sharpened, method="otsu")
            deskewed = cls.correct_perspective(thresholded)
            return deskewed, {"strategy": "standard", "steps": ["resize", "grayscale", "bilateral", "clahe", "sharpen", "otsu", "deskew"]}
