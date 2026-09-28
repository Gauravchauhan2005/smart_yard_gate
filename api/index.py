"""
Vercel Serverless Function entry point for Smart Yard Gate Automation System.
Exposes WSGI application object for Vercel Python runtime.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Mark serverless environment
os.environ.setdefault("VERCEL", "1")
os.environ.setdefault("FLASK_ENV", "production")

from app import create_app

# WSGI application callable for Vercel
app = create_app("production")
