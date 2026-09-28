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
