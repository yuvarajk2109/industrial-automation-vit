"""
CaneNexus – Logs Route
GET /api/logs – Retrieve analysis logs from MongoDB.
GET /api/logs/<id> – Retrieve a single log by ID.
GET /api/simulations – Retrieve simulation session summaries.
"""

from flask import Blueprint, request, jsonify
from bson import ObjectId, json_util
import json

from database.mongo_client import logs_collection, simulations_collection

logs_bp = Blueprint("logs", __name__)


def _serialize_doc(doc):
    """Convert a MongoDB document to JSON-serializable dict."""
    return json.loads(json_util.dumps(doc))


@logs_bp.route("/logs", methods=["GET"])
def get_logs():
    """
    Retrieve paginated analysis logs.

    Query params:
        session_id  – Filter by simulation session
        domain      – Filter by domain (steel/sugar)
        page        – Page number (default 1)
        limit       – Items per page (default 20, max 100)
    """
    session_id = request.args.get("session_id", "")
    domain = request.args.get("domain", "")
    page = max(1, int(request.args.get("page", 1)))
    limit = min(100, max(1, int(request.args.get("limit", 20))))

    # Build query filter
    query = {}
    if session_id:
        query["session_id"] = session_id
    if domain and domain in ("steel", "sugar"):
        query["domain"] = domain

    # Execute query
    skip = (page - 1) * limit
    total = logs_collection.count_documents(query)

    pipeline = [
        {"$match": query},
        {"$sort": {"timestamp": -1}},
        {"$skip": skip},
        {"$limit": limit},
        {
            "$lookup": {
                "from": "feedback_corrections",
                "let": {"log_id_str": {"$toString": "$_id"}},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$log_id", "$$log_id_str"]},
                                    {"$eq": ["$status", "pending"]}
                                ]
                            }
                        }
                    },
                    {"$limit": 1}
                ],
                "as": "_pending_feedback"
            }
        },
        {
            "$addFields": {
                "has_pending_correction": {"$gt": [{"$size": "$_pending_feedback"}, 0]}
            }
        },
        {
            "$project": {
                "_pending_feedback": 0
            }
        }
    ]

    cursor = logs_collection.aggregate(pipeline)
    logs = [_serialize_doc(doc) for doc in cursor]

    return jsonify({
        "logs": logs,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit
    }), 200


@logs_bp.route("/logs/<log_id>", methods=["GET"])
def get_log_detail(log_id):
    """Retrieve a single log by its MongoDB ObjectId."""
    try:
        doc = logs_collection.find_one({"_id": ObjectId(log_id)})
    except Exception:
        return jsonify({"error": "Invalid log_id format"}), 400

    if not doc:
        return jsonify({"error": f"Log not found: {log_id}"}), 404

    return jsonify(_serialize_doc(doc)), 200


@logs_bp.route("/simulations", methods=["GET"])
def get_simulations():
    """Retrieve all simulation session summaries, newest first."""
    cursor = simulations_collection.find().sort("started_at", -1)
    simulations = [_serialize_doc(doc) for doc in cursor]

    return jsonify({"simulations": simulations}), 200

@logs_bp.route("/logs/stats", methods=["GET"])
def get_stats():
    """Retrieve aggregated stats for charts."""
    # 1. Total Domains Distribution (although frontend already has total counts, useful to wrap)
    domain_agg = list(logs_collection.aggregate([
        {"$group": {"_id": "$domain", "count": {"$sum": 1}}}
    ]))
    
    # 2. Steel Defect Distribution
    # The predicted_class or defect arrays might be inside model_prediction.defect_summary
    # Let's aggregate by standard decision or quality_assessment from Knowledge graph instead for a uniform bar chart.
    steel_agg = list(logs_collection.aggregate([
        {"$match": {"domain": "steel"}},
        {"$group": {"_id": "$knowledge_graph_output.decision", "count": {"$sum": 1}}}
    ]))
    
    # 3. Sugar Class Distribution
    sugar_agg = list(logs_collection.aggregate([
        {"$match": {"domain": "sugar"}},
        {"$group": {"_id": "$model_prediction.predicted_class", "count": {"$sum": 1}}}
    ]))
    
    # 4. Performance Times (last 20 logs)
    cursor = logs_collection.find({}, {"processing_time_ms": 1, "timestamp": 1, "domain": 1}).sort("timestamp", -1).limit(20)
    perf_logs = [_serialize_doc(doc) for doc in cursor]
    perf_logs.reverse() # chronological order
    
    return jsonify({
        "domains": domain_agg,
        "steel_decisions": steel_agg,
        "sugar_classes": sugar_agg,
        "performance": perf_logs
    }), 200
