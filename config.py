"""
Configuration module for AI-Based Smart Yard Gate Automation System.
Defines environment-specific settings using clean object-oriented inheritance.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the smart-yard-gate project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file if it exists
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base configuration with shared defaults across environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "default-dev-secret-key-384918237")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))

    # File Upload Configuration
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "static" / "uploads"))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB default
    ALLOWED_EXTENSIONS = set(
        os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,mp4,avi,mov").split(",")
    )

    # Database Configuration
    USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() in ("true", "1", "yes")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_NAME = os.getenv("DB_NAME", "smart_yard_db")
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "smart_yard.db"))

    @classmethod
    def get_database_uri(cls) -> str:
        """Construct database URI based on active environment flags."""
        if cls.USE_SQLITE or not cls.DB_PASSWORD:
            return f"sqlite:///{cls.SQLITE_DB_PATH}"
        return (
            f"mysql+pymysql://{cls.DB_USER}:{cls.DB_PASSWORD}"
            f"@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return self.get_database_uri()

    # YOLO & Computer Vision Configuration
    YOLO_MODEL_PATH = os.getenv(
        "YOLO_MODEL_PATH", str(BASE_DIR / "models" / "yolo" / "yolov8n.pt")
    )
    DETECTION_CONF_THRESHOLD = float(os.getenv("DETECTION_CONF_THRESHOLD", 0.50))

    # OCR Configuration
    OCR_CONF_THRESHOLD = float(os.getenv("OCR_CONF_THRESHOLD", 0.60))


class DevelopmentConfig(Config):
    """Configuration for local development."""

    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Configuration for automated test execution."""

    DEBUG = False
    TESTING = True
    USE_SQLITE = True
    SQLITE_DB_PATH = ":memory:"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Configuration for production deployment."""

    DEBUG = False
    TESTING = False

    @classmethod
    def get_database_uri(cls) -> str:
        """Production requires configured database credentials."""
        if cls.USE_SQLITE:
            return f"sqlite:///{cls.SQLITE_DB_PATH}"
        return (
            f"mysql+pymysql://{cls.DB_USER}:{cls.DB_PASSWORD}"
            f"@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
        )


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
