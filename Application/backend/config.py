"""
CaneNexus – Centralised Configuration
All paths, DB URIs, model paths, API keys, and constants.
"""

import os
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent  # Project root

STEEL_MODEL_PATH = ROOT_DIR / "Models" / "Final" / "steel.pth"
SUGAR_MODEL_PATH = ROOT_DIR / "Models" / "Final" / "sugar.pth"

STEEL_TEST_IMAGES = ROOT_DIR / "Datasets" / "steel-defect-detection" / "test_images"
SUGAR_TEST_IMAGES = ROOT_DIR / "Datasets" / "sugar-quality-inspection" / "test_images"

OUTPUT_DIR = BASE_DIR / "static" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MongoDB ──
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "canenexus"

# ── Gemini ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# ── Model Config ──
STEEL_CLASSES = ["1", "2", "3", "4"]
SUGAR_CLASSES = ["unsaturated", "metastable", "intermediate", "labile"]

NUM_STEEL_CLASSES = 4
NUM_SUGAR_CLASSES = 4

STEEL_IMAGE_SIZE = 256
SUGAR_IMAGE_SIZE = 224

# ── Simulation Defaults ──
DEFAULT_SIMULATION_LIMIT = 50  # per domain

# ── Allowed Image Extensions ──
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
