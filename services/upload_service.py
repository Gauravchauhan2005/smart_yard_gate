"""
Upload and validation service for AI-Based Smart Yard Gate Automation System.
Provides secure multi-layer validation, content verification, and disk persistence
for gate camera still frames and video streams.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any, BinaryIO
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from PIL import Image
import cv2
import numpy as np


class UploadValidationError(Exception):
    """Base exception for upload and validation failures."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class MissingFileError(UploadValidationError):
    """Raised when no file was provided in the upload request."""
    pass


class InvalidExtensionError(UploadValidationError):
    """Raised when file extension is not permitted."""
    pass


class CorruptImageError(UploadValidationError):
    """Raised when image bytes cannot be decoded or verified."""
    pass


class FileSizeExceededError(UploadValidationError):
    """Raised when payload exceeds size limit."""

    def __init__(self, message: str = "File exceeds maximum permitted size."):
        super().__init__(message, status_code=413)


class UploadService:
    """
    Handles secure validation, deep content verification, and storage
    of gate vehicle images and videos.
    """

    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
    DEFAULT_MAX_BYTES = 16 * 1024 * 1024  # 16 MB

    # Magic byte signatures for video validation
    VIDEO_SIGNATURES = [
        b"ftyp",  # MP4 / MOV
        b"RIFF",  # AVI
        b"\x1a\x45\xdf\xa3",  # MKV / WebM
        b"\x00\x00\x00\x18ftyp",
        b"\x00\x00\x00\x20ftyp",
    ]

    def __init__(
        self,
        upload_folder: str | Path | None = None,
        max_content_length: int | None = None,
    ):
        if upload_folder:
            self.upload_folder = Path(upload_folder)
        else:
            self.upload_folder = Path(__file__).resolve().parent.parent / "static" / "uploads"

        self.max_content_length = max_content_length or self.DEFAULT_MAX_BYTES
        os.makedirs(self.upload_folder, exist_ok=True)

    @classmethod
    def get_file_extension(cls, filename: str) -> str:
        """Extract lowercase extension without leading dot."""
        if not filename or "." not in filename:
            return ""
        return filename.rsplit(".", 1)[1].lower()

    @classmethod
    def is_allowed_file(cls, filename: str) -> Tuple[bool, str]:
        """
        Validates if filename has an allowed extension.
        Returns (is_allowed, media_type) where media_type is 'image' or 'video'.
        """
        ext = cls.get_file_extension(filename)
        if ext in cls.ALLOWED_IMAGE_EXTENSIONS:
            return True, "image"
        if ext in cls.ALLOWED_VIDEO_EXTENSIONS:
            return True, "video"
        return False, ""

    def validate_image_content(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Deep validation of image bytes:
        - PIL verify to detect malformed headers
        - PIL decode to detect truncated bitstreams
        - OpenCV decodability check to guarantee compatibility with vision pipeline
        """
        import io

        if len(file_bytes) == 0:
            raise CorruptImageError("Uploaded file is 0 bytes.")

        # 1. PIL verification
        try:
            byte_stream = io.BytesIO(file_bytes)
            with Image.open(byte_stream) as img:
                img.verify()
        except Exception as e:
            raise CorruptImageError(f"Image integrity verification failed: {e}")

        # 2. PIL decoding & dimension inspection
        try:
            byte_stream.seek(0)
            with Image.open(byte_stream) as img:
                width, height = img.size
                img_format = img.format
                img_mode = img.mode

                if width < 32 or height < 32:
                    raise CorruptImageError(
                        f"Image resolution {width}x{height} is too small for vehicle/plate detection (min 32x32)."
                    )
                if width > 8192 or height > 8192:
                    raise CorruptImageError(
                        f"Image resolution {width}x{height} exceeds maximum supported dimensions (8192x8192)."
                    )
        except CorruptImageError:
            raise
        except Exception as e:
            raise CorruptImageError(f"Failed to read image attributes: {e}")

        # 3. OpenCV decodability verification
        nparr = np.frombuffer(file_bytes, np.uint8)
        cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise CorruptImageError("OpenCV could not decode image bytes into valid pixel matrix.")

        channels = cv_img.shape[2] if len(cv_img.shape) > 2 else 1

        return {
            "width": width,
            "height": height,
            "format": img_format,
            "mode": img_mode,
            "channels": channels,
        }

    def validate_video_content(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Validates video header signatures to prevent malicious payload uploads.
        """
        if len(file_bytes) < 32:
            raise CorruptImageError("Video file size is too small to contain valid container header.")

        header = file_bytes[:64]
        has_valid_signature = any(sig in header for sig in self.VIDEO_SIGNATURES)

        if not has_valid_signature:
            raise CorruptImageError("Uploaded video does not match supported container signatures (MP4, AVI, MOV).")

        return {
            "format": "video",
            "bytes_inspected": len(header),
        }

    def save_file(
        self,
        file: FileStorage,
        subfolder: str | None = None,
    ) -> Dict[str, Any]:
        """
        Validates, sanitizes, and writes uploaded file to disk.
        Returns structured metadata for the saved file.
        """
        if not file or not file.filename:
            raise MissingFileError("No file was selected or uploaded.")

        filename = file.filename
        allowed, media_type = self.is_allowed_file(filename)
        if not allowed:
            ext = self.get_file_extension(filename)
            allowed_list = sorted(list(self.ALLOWED_IMAGE_EXTENSIONS | self.ALLOWED_VIDEO_EXTENSIONS))
            raise InvalidExtensionError(
                f"File format '.{ext}' is not supported. Permitted formats: {', '.join(allowed_list)}"
            )

        # Read file into memory for content validation
        file.seek(0)
        file_bytes = file.read()
        file_size = len(file_bytes)

        if file_size > self.max_content_length:
            raise FileSizeExceededError(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed ({self.max_content_length / (1024 * 1024):.1f} MB)."
            )

        # Perform deep content validation
        if media_type == "image":
            content_meta = self.validate_image_content(file_bytes)
        else:
            content_meta = self.validate_video_content(file_bytes)

        # Generate unique, collision-free, path-safe filename
        safe_base = secure_filename(filename)
        if not safe_base:
            safe_base = f"upload_{uuid.uuid4().hex[:8]}.{self.get_file_extension(filename)}"

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_token = uuid.uuid4().hex[:8]
        saved_filename = f"{timestamp_str}_{unique_token}_{safe_base}"

        # Target directory
        target_dir = self.upload_folder
        if subfolder:
            target_dir = target_dir / subfolder
            os.makedirs(target_dir, exist_ok=True)

        destination_path = target_dir / saved_filename

        # Write validated bytes to disk
        with open(destination_path, "wb") as f:
            f.write(file_bytes)

        # Build relative path for web serving
        relative_path = f"static/uploads/{subfolder + '/' if subfolder else ''}{saved_filename}"

        result = {
            "original_filename": filename,
            "saved_filename": saved_filename,
            "file_path": relative_path,
            "absolute_path": str(destination_path),
            "media_type": media_type,
            "file_size": file_size,
            "status": "validated",
        }
        result.update(content_meta)
        return result
