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

    IS_VERCEL = os.getenv("VERCEL") == "1"

    # File Upload Configuration
    UPLOAD_FOLDER = os.getenv(
        "UPLOAD_FOLDER",
        "/tmp/uploads" if IS_VERCEL else str(BASE_DIR / "static" / "uploads"),
    )
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB default
    ALLOWED_EXTENSIONS = set(
        os.getenv("ALLOWED_EXTENSIONS", "png,jpg,jpeg,mp4,avi,mov").split(",")
    )

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL")
    USE_SQLITE = os.getenv("USE_SQLITE", "true" if IS_VERCEL else "false").lower() in ("true", "1", "yes")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_NAME = os.getenv("DB_NAME", "smart_yard_db")
    SQLITE_DB_PATH = os.getenv(
        "SQLITE_DB_PATH",
        "/tmp/smart_yard.db" if IS_VERCEL else str(BASE_DIR / "smart_yard.db"),
    )

    @staticmethod
    def build_database_uri(
        use_sqlite: bool,
        db_user: str,
        db_password: str,
        db_host: str,
        db_port: str,
        db_name: str,
        sqlite_path: str,
    ) -> str:
        """Construct database URI based on active environment flags."""
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            if db_url.startswith("postgres://"):
                db_url = db_url.replace("postgres://", "postgresql://", 1)
            return db_url

        if use_sqlite or not db_password:
            return f"sqlite:///{sqlite_path}"
        return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = build_database_uri(
        USE_SQLITE, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, SQLITE_DB_PATH
    )

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

    SQLALCHEMY_DATABASE_URI = Config.build_database_uri(
        Config.USE_SQLITE,
        Config.DB_USER,
        Config.DB_PASSWORD,
        Config.DB_HOST,
        Config.DB_PORT,
        Config.DB_NAME,
        Config.SQLITE_DB_PATH,
    )


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
