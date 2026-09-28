"""
Routes package initialization.
Exports all application blueprints for registration in the Flask app factory.
"""

from .main import main_bp
from .gate import gate_bp
from .api import api_bp

__all__ = ["main_bp", "gate_bp", "api_bp"]
