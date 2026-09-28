"""
Gate automation routes blueprint.
Handles gate camera feed, vehicle detection workflow, and check-in/out UI.
"""

from flask import Blueprint, render_template

gate_bp = Blueprint("gate", __name__, url_prefix="/gate")


@gate_bp.route("/")
def gate_index():
    """Renders the gate automation and visual inspection terminal."""
    return render_template("gate.html", active_page="gate")
