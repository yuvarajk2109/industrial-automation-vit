"""
CaneNexus – Flask Application Entry Point
Registers all blueprints, CORS, and pre-loads the model on startup.
"""

from flask import Flask, jsonify
from flask_cors import CORS

from config import OUTPUT_DIR

app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# - Register Blueprints -
from routes.predict import predict_bp
from routes.simulate import simulate_bp
from routes.chat import chat_bp
from routes.logs import logs_bp
from routes.images import images_bp
from routes.browse import browse_bp
from routes.feedback import feedback_bp
from routes.finetune import finetune_bp

app.register_blueprint(predict_bp, url_prefix="/api")
app.register_blueprint(simulate_bp, url_prefix="/api")
app.register_blueprint(chat_bp, url_prefix="/api")
app.register_blueprint(logs_bp, url_prefix="/api")
app.register_blueprint(images_bp, url_prefix="/api")
app.register_blueprint(browse_bp, url_prefix="/api")
app.register_blueprint(feedback_bp, url_prefix="/api")
app.register_blueprint(finetune_bp, url_prefix="/api")


# - Health Check -
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint – returns system status."""
    from models.loader import get_device
    from database.mongo_client import check_connection

    return jsonify({
        "status": "ok",
        "service": "CaneNexus Backend",
        "device": str(get_device()),
        "mongodb": check_connection()
    }), 200


# - Error Handlers -
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == "__main__":
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load the DDA-ViT model on startup
    print("[CaneNexus] Starting backend server...")
    try:
        from models.loader import get_model
        get_model()
    except Exception as e:
        print(f"[CaneNexus] WARNING: Model pre-load failed: {e}")
        print("[CaneNexus] The model will attempt to load on first request.")

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
