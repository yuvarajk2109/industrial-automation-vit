"""
CaneNexus – Simulate Route
POST /api/simulate – Virtual simulation with SSE streaming.
"""

import os
from flask import Blueprint, request, Response

from services.simulation import run_simulation_stream
from config import DEFAULT_SIMULATION_LIMIT

simulate_bp = Blueprint("simulate", __name__)


@simulate_bp.route("/simulate", methods=["POST"])
def simulate():
    """
    Start a virtual simulation that processes images from two directories.

    Accepts JSON body:
        {
            "steel_dir": "/path/to/steel/images",
            "sugar_dir": "/path/to/sugar/images",
            "limit": 50  (optional, default from config)
        }

    Returns:
        SSE event stream with real-time pipeline updates.
    """
    data = request.get_json()

    if not data:
        return {"error": "Request body must be JSON"}, 400

    steel_dir = data.get("steel_dir", "").strip()
    sugar_dir = data.get("sugar_dir", "").strip()
    limit = data.get("limit", DEFAULT_SIMULATION_LIMIT)

    # ── Validation ──
    if not steel_dir or not sugar_dir:
        return {"error": "Both steel_dir and sugar_dir are required"}, 400

    if not os.path.isdir(steel_dir):
        return {"error": f"Steel directory not found: {steel_dir}"}, 404

    if not os.path.isdir(sugar_dir):
        return {"error": f"Sugar directory not found: {sugar_dir}"}, 404

    # ── Stream SSE events ──
    return Response(
        run_simulation_stream(steel_dir, sugar_dir, limit),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "http://localhost:4200"
        }
    )
