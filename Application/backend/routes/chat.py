"""
CaneNexus – Chat Route
POST /api/chat – Gemini chatbot conversation endpoint.
"""

from datetime import datetime
from flask import Blueprint, request, jsonify
from bson import ObjectId

from chatbot.gemini_client import chat_response
from database.mongo_client import logs_collection, chats_collection

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    """
    Send a chat message in the context of a specific analysis.

    Accepts JSON body:
        {
            "log_id": "MongoDB ObjectId string",
            "message": "User's question"
        }

    Returns:
        {"response": "Gemini's reply", "history": [...]}
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    log_id = data.get("log_id", "").strip()
    message = data.get("message", "").strip()

    if not log_id or not message:
        return jsonify({"error": "Both log_id and message are required"}), 400

    # ── Retrieve analysis context from logs ──
    try:
        log_doc = logs_collection.find_one({"_id": ObjectId(log_id)})
    except Exception:
        return jsonify({"error": "Invalid log_id format"}), 400

    if not log_doc:
        return jsonify({"error": f"No log found with id: {log_id}"}), 404

    prediction = log_doc.get("model_prediction", {})
    kg_result = log_doc.get("knowledge_graph_output", {})

    # ── Retrieve or create chat document ──
    chat_doc = chats_collection.find_one({"log_id": log_id})

    if not chat_doc:
        chat_doc = {
            "log_id": log_id,
            "session_id": log_doc.get("session_id", ""),
            "messages": []
        }
        chats_collection.insert_one(chat_doc)

    # ── Build history for Gemini ──
    history = []
    for msg in chat_doc.get("messages", []):
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # ── Get Gemini response ──
    try:
        response_text = chat_response(
            history=history,
            user_message=message,
            prediction_result=prediction,
            kg_result=kg_result
        )
    except Exception as e:
        return jsonify({"error": f"Gemini chat failed: {str(e)}"}), 500

    # ── Save messages to chat document ──
    now = datetime.utcnow()
    new_messages = [
        {"role": "user", "content": message, "timestamp": now},
        {"role": "model", "content": response_text, "timestamp": now}
    ]

    chats_collection.update_one(
        {"log_id": log_id},
        {"$push": {"messages": {"$each": new_messages}}}
    )

    # ── Return response ──
    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "model", "content": response_text}
    ]

    return jsonify({
        "response": response_text,
        "history": updated_history
    }), 200
