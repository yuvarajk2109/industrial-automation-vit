"""
CaneNexus – Predict Route
POST /api/predict – Single image analysis endpoint.
"""

import os
from flask import Blueprint, request, jsonify

from services.pipeline import run_pipeline

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Run the full analysis pipeline on a single image.

    Accepts JSON body:
        {
            "image_path": "/absolute/path/to/image.jpg",
            "domain": "steel" | "sugar"
        }

    Returns:
        Full pipeline result including prediction, KG output, and Gemini response.
    """
    # If client is sending a multi-part form (drag/drop upload)
    if "image" in request.files:
        file = request.files["image"]
        domain = request.form.get("domain", "").strip().lower()
        
        if not file or not file.filename:
            return jsonify({"error": "No file selected"}), 400
            
        # Temp save for inference loop
        from config import OUTPUT_DIR
        upload_dir = OUTPUT_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        image_path = str(upload_dir / file.filename)
        file.save(image_path)
    else:
        # Fallback to JSON payload path 
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request must be JSON or contain a file"}), 400
        image_path = data.get("image_path", "").strip()
        domain = data.get("domain", "").strip().lower()

    # ── Validation ──
    if not image_path:
        return jsonify({"error": "image_path is required"}), 400

    if domain not in ("steel", "sugar"):
        return jsonify({"error": "domain must be 'steel' or 'sugar'"}), 400

    if not os.path.isfile(image_path):
        return jsonify({"error": f"Image file not found: {image_path}"}), 404

    # ── Run Pipeline ──
    try:
        result = run_pipeline(image_path, domain)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": f"Pipeline failed: {str(e)}",
            "image_path": image_path,
            "domain": domain
        }), 500
