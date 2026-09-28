"""
AI-Based Smart Yard Gate Automation System - Application Entry Point.
Implements the Application Factory pattern for modularity, testability, and scalability.
"""

import os
import logging
from flask import Flask, jsonify
from config import config_by_name, BASE_DIR
from database.db import db


def configure_logging(app: Flask) -> None:
    """Set up standardized logging format and handlers."""
    log_level = logging.DEBUG if app.config.get("DEBUG", False) else logging.INFO
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    app.logger.setLevel(log_level)
    app.logger.info(f"Logging initialized at level: {logging.getLevelName(log_level)}")


def register_blueprints(app: Flask) -> None:
    """Register all application blueprints."""
    from routes.main import main_bp
    from routes.gate import gate_bp
    from routes.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(gate_bp)
    app.register_blueprint(api_bp)
    app.logger.info("Application blueprints registered successfully.")


def register_error_handlers(app: Flask) -> None:
    """Register uniform JSON error handlers across the application."""

    @app.errorhandler(400)
    def bad_request_error(error):
        return jsonify({
            "error": "Bad Request",
            "message": getattr(error, "description", "The request could not be processed due to invalid syntax."),
            "status_code": 400,
        }), 400

    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            "error": "Not Found",
            "message": getattr(error, "description", "The requested resource was not found on this server."),
            "status_code": 404,
        }), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "error": "Payload Too Large",
            "message": "The uploaded file exceeds the configured maximum upload size.",
            "status_code": 413,
        }), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f"Internal server error: {error}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred. Please contact the administrator.",
            "status_code": 500,
        }), 500


def create_app(config_name: str | None = None) -> Flask:
    """
    Application factory function.
    Initializes Flask application with configured environment, blueprints, and error handlers.
    """
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development").lower()

    app = Flask(__name__)

    # Load configuration
    config_class = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_class)

    # Ensure upload directory exists
    upload_folder = app.config.get("UPLOAD_FOLDER", str(BASE_DIR / "static" / "uploads"))
    os.makedirs(upload_folder, exist_ok=True)

    # Setup logging
    configure_logging(app)

    # Initialize database
    db.init_app(app)

    # Register blueprints and handlers
    register_blueprints(app)
    register_error_handlers(app)

    app.logger.info(f"Smart Yard Gate Application created in '{config_name}' mode.")
    return app


# Application instance for WSGI servers / flask run
app = create_app()


if __name__ == "__main__":
    host = app.config.get("HOST", "0.0.0.0")
    port = app.config.get("PORT", 5000)
    debug = app.config.get("DEBUG", True)

    app.logger.info(f"Starting Smart Yard Gate Server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
