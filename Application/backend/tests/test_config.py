"""
Config Tests
    - Validates 
        - configuration constants
        - paths
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    STEEL_CLASSES, SUGAR_CLASSES,
    NUM_STEEL_CLASSES, NUM_SUGAR_CLASSES,
    STEEL_IMAGE_SIZE, SUGAR_IMAGE_SIZE,
    MONGO_URI, MONGO_DB, GEMINI_MODEL,
    ALLOWED_EXTENSIONS, DEFAULT_SIMULATION_LIMIT,
    BASE_DIR, ROOT_DIR, OUTPUT_DIR,
    STEEL_MODEL_PATH, SUGAR_MODEL_PATH
)


class TestModelConfig:
    """
    - Tests for model-related configuration
    """

    def test_steel_classes_count(self):
        assert len(STEEL_CLASSES) == NUM_STEEL_CLASSES == 4

    def test_sugar_classes_count(self):
        assert len(SUGAR_CLASSES) == NUM_SUGAR_CLASSES == 4

    def test_sugar_classes_names(self):
        expected = ["unsaturated", "metastable", "intermediate", "labile"]
        assert SUGAR_CLASSES == expected

    def test_steel_image_size(self):
        assert STEEL_IMAGE_SIZE == 256

    def test_sugar_image_size(self):
        assert SUGAR_IMAGE_SIZE == 224

    def test_gemini_model_name(self):
        assert GEMINI_MODEL == "gemini-2.5-flash"


class TestDatabaseConfig:
    """
    - Tests for MongoDB configuration
    """

    def test_mongo_uri_format(self):
        assert MONGO_URI.startswith("mongodb://")

    def test_mongo_db_name(self):
        assert MONGO_DB == "canenexus"


class TestPathConfig:
    """
    - Tests for path configuration
    """

    def test_base_dir_exists(self):
        assert BASE_DIR.exists()

    def test_output_dir_exists(self):
        # OUTPUT_DIR is created on import
        assert OUTPUT_DIR.exists()

    def test_model_paths_have_pth_extension(self):
        assert str(STEEL_MODEL_PATH).endswith(".pth")
        assert str(SUGAR_MODEL_PATH).endswith(".pth")

    def test_root_dir_is_project_root(self):
        # ROOT_DIR should be the "Project" directory
        assert ROOT_DIR.name == "Project" or ROOT_DIR.exists()


class TestSimulationConfig:
    """
    - Tests for simulation configuration
    """

    def test_default_limit_is_reasonable(self):
        assert 1 <= DEFAULT_SIMULATION_LIMIT <= 500

    def test_allowed_extensions_include_common_formats(self):
        assert ".jpg" in ALLOWED_EXTENSIONS
        assert ".png" in ALLOWED_EXTENSIONS
        assert ".jpeg" in ALLOWED_EXTENSIONS

    def test_allowed_extensions_are_lowercase(self):
        for ext in ALLOWED_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")