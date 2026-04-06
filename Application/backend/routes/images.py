"""
CaneNexus – Images Route
GET /api/images/<filename> – Serve generated output images (overlays, masks).
GET /api/source-image – Serve original source images by path.
"""

import os
from flask import Blueprint, send_from_directory, request, jsonify, send_file

from config import OUTPUT_DIR

images_bp = Blueprint("images", __name__)


@images_bp.route("/images/<filename>", methods=["GET"])
def serve_output_image(filename):
    """Serve a generated image from the static/outputs directory."""
    file_path = os.path.join(str(OUTPUT_DIR), filename)

    if not os.path.isfile(file_path):
        return jsonify({"error": f"Image not found: {filename}"}), 404

    return send_from_directory(str(OUTPUT_DIR), filename)


@images_bp.route("/source-image", methods=["GET"])
def serve_source_image():
    """
    Serve an original source image by its absolute path.
    Used by the frontend to display input images.

    Query params:
        path – Absolute path to the source image
    """
    image_path = request.args.get("path", "").strip()

    if not image_path:
        return jsonify({"error": "path query parameter is required"}), 400

    if not os.path.isfile(image_path):
        return jsonify({"error": f"Image not found: {image_path}"}), 404

    return send_file(image_path)
