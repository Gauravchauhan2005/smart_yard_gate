"""
Database package for Smart Yard Gate Automation System.
Exports SQLAlchemy database instance and ORM models.
"""

from .db import db
from .models import Vehicle, Gate, YardLocation, Detection

__all__ = ["db", "Vehicle", "Gate", "YardLocation", "Detection"]
