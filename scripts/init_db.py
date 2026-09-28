"""
Database initialization and migration script.
Creates tables and seeds default gate entries.
(Full implementation scheduled for Phase 3).
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def init_database():
    """Initializes the yard database and seeds default gate data."""
    print("Database initialization scheduled for Phase 3.")


if __name__ == "__main__":
    init_database()
