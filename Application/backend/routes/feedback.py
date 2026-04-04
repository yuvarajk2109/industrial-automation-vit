"""
CaneNexus – Feedback Route
POST /api/feedback          — Submit a single correction
POST /api/feedback/batch    — Submit batch corrections from simulation review
GET  /api/feedback          — List feedback entries (paginated, filtered)
GET  /api/feedback/stats    — Feedback statistics
"""

from flask import Blueprint, request, jsonify
from bson import ObjectId

from database.mongo_client import (
    feedback_collection, logs_collection
)
from database.schemas import create_feedback_document

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/feedback", methods=["POST"])
def submit_feedback():
    """
    Submit a single correction for a misclassified image.

    JSON body:
        {
            "log_id": "MongoDB ObjectId string",
            "domain": "sugar" | "steel",
            "corrected_label": { ... },   # domain-specific
            "reason": "(optional) text"
        }

    Returns:
        { "feedback_id": "...", "pending_count": int }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400

    log_id = data.get("log_id", "").strip()
    domain = data.get("domain", "").strip().lower()
    corrected_label = data.get("corrected_label")
    reason = data.get("reason", "").strip()

    # ── Validation ──
    if not log_id:
        return jsonify({"error": "log_id is required"}), 400
    if domain not in ("sugar", "steel"):
        return jsonify({"error": "domain must be 'sugar' or 'steel'"}), 400
    if not corrected_label:
        return jsonify({"error": "corrected_label is required"}), 400

    # Validate sugar correction
    if domain == "sugar":
        valid_classes = ["unsaturated", "metastable", "intermediate", "labile"]
        cls = corrected_label.get("class", "")
        if cls not in valid_classes:
            return jsonify({
                "error": f"corrected_label.class must be one of: {valid_classes}"
            }), 400

    # Validate steel correction
    if domain == "steel":
        if corrected_label.get("type") != "region_override":
            return jsonify({
                "error": "Steel corrected_label must have type 'region_override'"
            }), 400

    # ── Fetch original log to get image info ──
    try:
        log_doc = logs_collection.find_one({"_id": ObjectId(log_id)})
    except Exception:
        return jsonify({"error": "Invalid log_id format"}), 400

    if not log_doc:
        return jsonify({"error": f"Log entry not found: {log_id}"}), 404

    # ── Create and insert feedback ──
    feedback_doc = create_feedback_document(
        log_id=log_id,
        image_path=log_doc.get("image_path", ""),
        image_filename=log_doc.get("image_filename", ""),
        domain=domain,
        original_prediction=log_doc.get("model_prediction", {}),
        corrected_label=corrected_label,
        reason=reason,
        source="single_analysis"
    )

    try:
        result = feedback_collection.insert_one(feedback_doc)
        feedback_id = str(result.inserted_id)
    except Exception as e:
        return jsonify({"error": f"Failed to save feedback: {str(e)}"}), 500

    # Count pending corrections for this domain
    pending_count = feedback_collection.count_documents({
        "domain": domain,
        "status": "pending"
    })

    return jsonify({
        "feedback_id": feedback_id,
        "pending_count": pending_count
    }), 201


@feedback_bp.route("/feedback/batch", methods=["POST"])
def submit_batch_feedback():
    """
    Submit batch corrections from simulation review.

    JSON body:
        {
            "session_id": "uuid-of-simulation",
            "corrections": [
                {
                    "log_id": "...",
                    "domain": "sugar" | "steel",
                    "corrected_label": { ... },
                    "reason": "(optional)"
                },
                ...
            ]
        }

    Returns:
        { "submitted_count": int, "pending_count": { "sugar": int, "steel": int } }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400

    corrections = data.get("corrections", [])
    if not corrections:
        return jsonify({"error": "corrections array is required and must not be empty"}), 400

    submitted = 0
    errors = []

    for i, corr in enumerate(corrections):
        log_id = corr.get("log_id", "").strip()
        domain = corr.get("domain", "").strip().lower()
        corrected_label = corr.get("corrected_label")
        reason = corr.get("reason", "").strip()

        if not log_id or domain not in ("sugar", "steel") or not corrected_label:
            errors.append({"index": i, "error": "Missing required fields"})
            continue

        try:
            log_doc = logs_collection.find_one({"_id": ObjectId(log_id)})
        except Exception:
            errors.append({"index": i, "error": f"Invalid log_id: {log_id}"})
            continue

        if not log_doc:
            errors.append({"index": i, "error": f"Log not found: {log_id}"})
            continue

        feedback_doc = create_feedback_document(
            log_id=log_id,
            image_path=log_doc.get("image_path", ""),
            image_filename=log_doc.get("image_filename", ""),
            domain=domain,
            original_prediction=log_doc.get("model_prediction", {}),
            corrected_label=corrected_label,
            reason=reason,
            source="simulation_review"
        )

        try:
            feedback_collection.insert_one(feedback_doc)
            submitted += 1
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    # Count pending per domain
    pending = {
        "sugar": feedback_collection.count_documents({"domain": "sugar", "status": "pending"}),
        "steel": feedback_collection.count_documents({"domain": "steel", "status": "pending"})
    }

    result = {
        "submitted_count": submitted,
        "pending_count": pending
    }
    if errors:
        result["errors"] = errors

    return jsonify(result), 201


@feedback_bp.route("/feedback", methods=["GET"])
def list_feedback():
    """
    List feedback entries with optional filters.

    Query params:
        domain  — "sugar" | "steel" (optional)
        status  — "pending" | "used" | "discarded" (optional)
        page    — page number (default 1)
        limit   — items per page (default 20)
    """
    domain = request.args.get("domain", "").strip().lower()
    status = request.args.get("status", "").strip().lower()
    page = max(int(request.args.get("page", 1)), 1)
    limit = min(max(int(request.args.get("limit", 20)), 1), 100)

    query = {}
    if domain in ("sugar", "steel"):
        query["domain"] = domain
    if status in ("pending", "used", "discarded"):
        query["status"] = status

    total = feedback_collection.count_documents(query)
    skip = (page - 1) * limit

    docs = list(
        feedback_collection.find(query)
        .sort("submitted_at", -1)
        .skip(skip)
        .limit(limit)
    )

    # Serialise ObjectId fields
    for doc in docs:
        doc["_id"] = str(doc["_id"])

    return jsonify({
        "feedback": docs,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit
    }), 200


@feedback_bp.route("/feedback/stats", methods=["GET"])
def feedback_stats():
    """
    Get feedback statistics.

    Returns:
        {
            "total": int,
            "pending": int,
            "used": int,
            "per_domain": {
                "sugar": { "pending": int, "used": int, "total": int },
                "steel": { "pending": int, "used": int, "total": int }
            }
        }
    """
    total = feedback_collection.count_documents({})
    pending = feedback_collection.count_documents({"status": "pending"})
    used = feedback_collection.count_documents({"status": "used"})

    per_domain = {}
    for d in ("sugar", "steel"):
        per_domain[d] = {
            "pending": feedback_collection.count_documents({"domain": d, "status": "pending"}),
            "used": feedback_collection.count_documents({"domain": d, "status": "used"}),
            "total": feedback_collection.count_documents({"domain": d})
        }

    return jsonify({
        "total": total,
        "pending": pending,
        "used": used,
        "per_domain": per_domain
    }), 200
