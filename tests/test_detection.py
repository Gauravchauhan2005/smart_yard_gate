"""
Unit tests for upload validation and image verification pipelines.
Verifies file extension restrictions, magic byte validation, corrupted image rejection,
and secure storage mechanics.
"""

import io
from pathlib import Path
import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage
from services import (
    UploadService,
    MissingFileError,
    InvalidExtensionError,
    CorruptImageError,
    FileSizeExceededError,
)


import shutil


@pytest.fixture
def upload_dir():
    """Temporary upload directory fixture isolated to test workspace."""
    dest = Path(__file__).resolve().parent / "tmp_test_uploads"
    dest.mkdir(parents=True, exist_ok=True)
    yield dest
    shutil.rmtree(dest, ignore_errors=True)


@pytest.fixture
def uploader(upload_dir):
    """Configured UploadService pointing to temporary folder."""
    return UploadService(upload_folder=upload_dir, max_content_length=2 * 1024 * 1024)


def create_test_image(width: int = 120, height: int = 80, fmt: str = "JPEG", color: str = "blue") -> io.BytesIO:
    """Helper creating in-memory valid test image byte stream."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


def test_allowed_extensions_logic():
    """Verify whitelist validation for allowed images and videos."""
    assert UploadService.is_allowed_file("truck.jpg") == (True, "image")
    assert UploadService.is_allowed_file("trailer.PNG") == (True, "image")
    assert UploadService.is_allowed_file("gate_feed.mp4") == (True, "video")
    assert UploadService.is_allowed_file("capture.avi") == (True, "video")

    assert UploadService.is_allowed_file("malicious.exe") == (False, "")
    assert UploadService.is_allowed_file("script.py") == (False, "")
    assert UploadService.is_allowed_file("payload.php") == (False, "")
    assert UploadService.is_allowed_file("no_extension") == (False, "")


def test_valid_image_save(uploader):
    """Verify valid image is inspected, decoded, and stored to disk."""
    buf = create_test_image(width=640, height=480, fmt="JPEG")
    file_storage = FileStorage(
        stream=buf,
        filename="gate_camera_capture.jpg",
        content_type="image/jpeg",
    )

    meta = uploader.save_file(file_storage, subfolder="gate_1")

    assert meta["status"] == "validated"
    assert meta["media_type"] == "image"
    assert meta["width"] == 640
    assert meta["height"] == 480
    assert meta["channels"] == 3
    assert Path(meta["absolute_path"]).exists()
    assert "gate_1" in meta["file_path"]


def test_missing_file_raises_error(uploader):
    """Verify error raised when file or filename is missing."""
    with pytest.raises(MissingFileError):
        uploader.save_file(None)

    empty_file = FileStorage(stream=io.BytesIO(), filename="")
    with pytest.raises(MissingFileError):
        uploader.save_file(empty_file)


def test_invalid_extension_raises_error(uploader):
    """Verify rejection of files with non-whitelisted extensions."""
    fake_payload = FileStorage(
        stream=io.BytesIO(b"binary payload"),
        filename="exploit.exe",
        content_type="application/octet-stream",
    )
    with pytest.raises(InvalidExtensionError) as exc:
        uploader.save_file(fake_payload)
    assert "not supported" in str(exc.value)


def test_corrupt_image_raises_error(uploader):
    """Verify rejection when file extension is .jpg but content is corrupted random bytes."""
    corrupted_data = b"NOT_A_REAL_IMAGE_BYTES_XYZ_12345"
    fake_image = FileStorage(
        stream=io.BytesIO(corrupted_data),
        filename="fake_truck.jpg",
        content_type="image/jpeg",
    )
    with pytest.raises(CorruptImageError) as exc:
        uploader.save_file(fake_image)
    assert "verification failed" in str(exc.value) or "could not decode" in str(exc.value)


def test_tiny_resolution_rejection(uploader):
    """Verify rejection of images smaller than 32x32."""
    buf = create_test_image(width=16, height=16, fmt="PNG")
    tiny_image = FileStorage(
        stream=buf,
        filename="tiny_thumbnail.png",
        content_type="image/png",
    )
    with pytest.raises(CorruptImageError) as exc:
        uploader.save_file(tiny_image)
    assert "too small" in str(exc.value)


def test_file_size_exceeded_error(upload_dir):
    """Verify FileSizeExceededError when payload exceeds threshold."""
    strict_uploader = UploadService(upload_folder=upload_dir, max_content_length=100)  # 100 bytes max
    buf = create_test_image(width=100, height=100, fmt="JPEG")
    large_image = FileStorage(
        stream=buf,
        filename="large.jpg",
        content_type="image/jpeg",
    )
    with pytest.raises(FileSizeExceededError):
        strict_uploader.save_file(large_image)


def test_valid_video_signature(uploader):
    """Verify video container signature validation."""
    valid_mp4_header = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42" + (b"\x00" * 64)
    video_storage = FileStorage(
        stream=io.BytesIO(valid_mp4_header),
        filename="gate_stream.mp4",
        content_type="video/mp4",
    )
    meta = uploader.save_file(video_storage)
    assert meta["status"] == "validated"
    assert meta["media_type"] == "video"
    assert Path(meta["absolute_path"]).exists()


def test_image_preprocessor_pipeline():
    """Verify OpenCV preprocessing steps and strategies."""
    import numpy as np
    from detection.preprocessing import ImagePreprocessor

    # Create dummy color image
    dummy_img = np.zeros((100, 250, 3), dtype=np.uint8)
    dummy_img[20:80, 20:230] = 200  # Plate rectangle

    # 1. Grayscale
    gray = ImagePreprocessor.to_grayscale(dummy_img)
    assert len(gray.shape) == 2

    # 2. Resize
    resized = ImagePreprocessor.resize_plate(dummy_img, target_height=64)
    assert resized.shape[0] == 64

    # 3. Crop
    cropped = ImagePreprocessor.crop_region(dummy_img, [20, 20, 230, 80], margin=2)
    assert cropped.size > 0

    # 4. Pipeline strategies
    out_std, meta_std = ImagePreprocessor.preprocess_pipeline(dummy_img, strategy="standard")
    assert out_std is not None
    assert meta_std["strategy"] == "standard"

    out_adapt, meta_adapt = ImagePreprocessor.preprocess_pipeline(dummy_img, strategy="adaptive")
    assert out_adapt is not None

    out_contrast, meta_contrast = ImagePreprocessor.preprocess_pipeline(dummy_img, strategy="contrast_boost")
    assert out_contrast is not None


def test_plate_detector_localization():
    """Verify plate detector extracts candidate bounding box and crop."""
    import numpy as np
    from detection.plate_detector import PlateDetector

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simulated rectangular plate
    dummy_frame[300:360, 220:420] = 255

    detector = PlateDetector()
    results = detector.detect_plates(dummy_frame)

    assert len(results) > 0
    plate = results[0]
    assert "bbox" in plate
    assert "confidence" in plate
    assert "crop" in plate
    assert plate["confidence"] > 0.0


def test_ocr_text_cleaning_and_heuristics():
    """Verify OCR character cleaning and Indian HSRP domain error corrections."""
    from detection.ocr import PlateOCR

    # 1. Clean punctuation and spaces
    raw1 = " [mh 12-rn-8842]  "
    clean1 = PlateOCR.clean_and_normalize_text(raw1)
    assert clean1 == "MH-12-RN-8842"

    # 2. Indian HSRP Number heuristics: formats SS-RR-LL-NNNN with character error correction
    raw2 = "MH-12-RN-8842"
    corrected2 = PlateOCR.correct_plate_heuristics("MH12RN8842")
    assert corrected2 == "MH-12-RN-8842"

    # 3. Disambiguate letter 'O' vs digit '0' in RTO code
    raw_ocr_misread = "MHO2RN8842"
    corrected3 = PlateOCR.correct_plate_heuristics(raw_ocr_misread)
    assert corrected3 == "MH-02-RN-8842"

    # 4. Disambiguate letter 'B' vs digit '8' in 4-digit registration
    raw_ocr_misread2 = "MH12RNB842"
    corrected4 = PlateOCR.correct_plate_heuristics(raw_ocr_misread2)
    assert corrected4 == "MH-12-RN-8842"

    # 5. Empty string
    assert PlateOCR.clean_and_normalize_text("") == ""


def test_ocr_service_recognition():
    """Verify OCRService runs extraction and returns structured payload."""
    import numpy as np
    from services.ocr_service import OCRService

    dummy_plate = np.ones((60, 200, 3), dtype=np.uint8) * 255
    ocr_svc = OCRService()
    res = ocr_svc.recognize_plate(dummy_plate)

    assert "license_plate" in res
    assert "trailer_number" in res
    assert "ocr_confidence" in res
    assert "needs_manual_review" in res


def test_detection_service_pipeline(upload_dir):
    """Verify DetectionService coordinates vehicle detection and visual artifacts."""
    import numpy as np
    from services.detection_service import DetectionService

    dummy_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    det_svc = DetectionService(output_dir=upload_dir)
    res = det_svc.process_frame(dummy_frame, gate_number=1, save_visuals=True)

    assert "vehicle" in res
    assert "plate" in res
    assert "visuals" in res
    assert "detected" in res["vehicle"]
    assert res["visuals"]["annotated_image"] is not None
