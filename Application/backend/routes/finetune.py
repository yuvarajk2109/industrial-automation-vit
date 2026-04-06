"""
CaneNexus – Fine-Tune Route
POST /api/finetune/start    — trigger a fine-tune job
GET  /api/finetune/status   — current job status
GET  /api/finetune/history  — all past jobs
POST /api/finetune/rollback — revert to a previous model version
GET  /api/finetune/versions — list model version checkpoints
"""

from flask import Blueprint, request, jsonify

from services.fine_tune.scheduler import (
    start_finetune_job, get_current_job, get_job_history
)
from services.fine_tune.model_manager import (
    rollback_model, list_versions, get_active_version
)

finetune_bp = Blueprint("finetune", __name__)


@finetune_bp.route("/finetune/start", methods=["POST"])
def start_finetune():
    """
    Trigger a fine-tune job for a specific domain.

    JSON body:
        {
            "domain": "sugar" | "steel",
            "config": {                     // optional overrides
                "lr": 1e-4,
                "epochs": 10,
                "min_corrections": 5,
                "validation_split": 0.2,
                "unfreeze_projection": false,
                "early_stopping_patience": 3
            }
        }

    Returns:
        { "job_id": "...", "status": "running", "message": "..." }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400

    domain = data.get("domain", "").strip().lower()
    config = data.get("config", {})

    if domain not in ("sugar", "steel"):
        return jsonify({"error": "domain must be 'sugar' or 'steel'"}), 400

    result = start_finetune_job(domain, config)

    if "error" in result:
        status_code = 409 if result.get("status") == "busy" else 400
        return jsonify(result), status_code

    return jsonify(result), 202


@finetune_bp.route("/finetune/status", methods=["GET"])
def finetune_status():
    """
    Get the current/most-recent fine-tune job status.

    Returns:
        Job status dict or {"status": "idle"} if no job.
    """
    job = get_current_job()
    if job:
        return jsonify(job), 200

    return jsonify({"status": "idle"}), 200


@finetune_bp.route("/finetune/history", methods=["GET"])
def finetune_history():
    """
    Get all past fine-tune jobs.

    Returns:
        List of job documents.
    """
    jobs = get_job_history()
    return jsonify({"jobs": jobs}), 200


@finetune_bp.route("/finetune/rollback", methods=["POST"])
def finetune_rollback():
    """
    Rollback the active model to a previous version.

    JSON body:
        {
            "domain": "sugar" | "steel",
            "version": int
        }

    Returns:
        { "status": "rolled_back", "domain": "...", "restored_version": int }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400

    domain = data.get("domain", "").strip().lower()
    version = data.get("version")

    if domain not in ("sugar", "steel"):
        return jsonify({"error": "domain must be 'sugar' or 'steel'"}), 400

    if not isinstance(version, int) or version < 1:
        return jsonify({"error": "version must be a positive integer"}), 400

    try:
        result = rollback_model(domain, version)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Rollback failed: {str(e)}"}), 500


@finetune_bp.route("/finetune/versions", methods=["GET"])
def model_versions():
    """
    List all model versions, optionally filtered by domain.

    Query params:
        domain — "sugar" | "steel" (optional)

    Returns:
        { "versions": [...], "active_versions": { "sugar": int, "steel": int } }
    """
    domain = request.args.get("domain", "").strip().lower()
    domain_filter = domain if domain in ("sugar", "steel") else None

    versions = list_versions(domain_filter)

    active = {
        "sugar": get_active_version("sugar"),
        "steel": get_active_version("steel")
    }

    return jsonify({
        "versions": versions,
        "active_versions": active
    }), 200
