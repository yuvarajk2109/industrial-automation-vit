"""
Shared Test Fixtures
    - Provides
        - mocked model
        - MongoDB
        - Gemini 
    for isolated unit testing
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def steel_prediction_no_defects():
    """
    - Steel prediction with no defects detected
    """
    return {
        "image_path": "E:/test/steel_001.jpg",
        "image_filename": "steel_001.jpg",
        "domain": "steel",
        "defect_summary": {
            "class_1": {"detected": False, "area_pct": 0.0},
            "class_2": {"detected": False, "area_pct": 0.0},
            "class_3": {"detected": False, "area_pct": 0.0},
            "class_4": {"detected": False, "area_pct": 0.0},
        },
        "dominant_defect": "none",
        "total_defect_area_pct": 0.0,
        "mask_overlay_path": "steel_001_abc12345_overlay.png",
        "raw_mask_path": "steel_001_abc12345_mask.png",
        "inference_time_ms": 150.0,
    }


@pytest.fixture
def steel_prediction_minor_defect():
    """
    - Steel prediction with a single minor defect
    """
    return {
        "image_path": "E:/test/steel_002.jpg",
        "image_filename": "steel_002.jpg",
        "domain": "steel",
        "defect_summary": {
            "class_1": {"detected": True, "area_pct": 0.5},
            "class_2": {"detected": False, "area_pct": 0.0},
            "class_3": {"detected": False, "area_pct": 0.0},
            "class_4": {"detected": False, "area_pct": 0.0},
        },
        "dominant_defect": "class_1",
        "total_defect_area_pct": 0.5,
        "mask_overlay_path": "steel_002_overlay.png",
        "raw_mask_path": "steel_002_mask.png",
        "inference_time_ms": 160.0,
    }


@pytest.fixture
def steel_prediction_severe_defect():
    """
    - Steel prediction with large defect area
    """
    return {
        "image_path": "E:/test/steel_003.jpg",
        "image_filename": "steel_003.jpg",
        "domain": "steel",
        "defect_summary": {
            "class_1": {"detected": True, "area_pct": 3.5},
            "class_2": {"detected": True, "area_pct": 2.0},
            "class_3": {"detected": False, "area_pct": 0.0},
            "class_4": {"detected": False, "area_pct": 0.0},
        },
        "dominant_defect": "class_1",
        "total_defect_area_pct": 5.5,
        "mask_overlay_path": "steel_003_overlay.png",
        "raw_mask_path": "steel_003_mask.png",
        "inference_time_ms": 170.0,
    }


@pytest.fixture
def steel_prediction_critical_defect():
    """
    - Steel prediction with class 4 (elongated) + high area
    - Triggers manual inspection
    """
    return {
        "image_path": "E:/test/steel_004.jpg",
        "image_filename": "steel_004.jpg",
        "domain": "steel",
        "defect_summary": {
            "class_1": {"detected": True, "area_pct": 2.0},
            "class_2": {"detected": False, "area_pct": 0.0},
            "class_3": {"detected": False, "area_pct": 0.0},
            "class_4": {"detected": True, "area_pct": 5.0},
        },
        "dominant_defect": "class_4",
        "total_defect_area_pct": 7.0,
        "mask_overlay_path": "steel_004_overlay.png",
        "raw_mask_path": "steel_004_mask.png",
        "inference_time_ms": 180.0,
    }


@pytest.fixture
def steel_prediction_widespread():
    """
    - Steel prediction with 3+ classes detected
    - Triggers widespread pattern
    """
    return {
        "image_path": "E:/test/steel_005.jpg",
        "image_filename": "steel_005.jpg",
        "domain": "steel",
        "defect_summary": {
            "class_1": {"detected": True, "area_pct": 1.0},
            "class_2": {"detected": True, "area_pct": 0.8},
            "class_3": {"detected": True, "area_pct": 0.5},
            "class_4": {"detected": False, "area_pct": 0.0},
        },
        "dominant_defect": "class_1",
        "total_defect_area_pct": 2.3,
        "mask_overlay_path": "steel_005_overlay.png",
        "raw_mask_path": "steel_005_mask.png",
        "inference_time_ms": 175.0,
    }


@pytest.fixture
def sugar_prediction_metastable():
    """
    - Sugar prediction
    - Metastable class
    """
    return {
        "image_path": "E:/test/sugar_001.jpg",
        "image_filename": "sugar_001.jpg",
        "domain": "sugar",
        "predicted_class": "metastable",
        "confidence": 0.92,
        "all_probabilities": {
            "unsaturated": 0.02,
            "metastable": 0.92,
            "intermediate": 0.04,
            "labile": 0.02,
        },
        "inference_time_ms": 80.0,
    }


@pytest.fixture
def sugar_prediction_labile():
    """
    - Sugar prediction
    - Labile class
    """
    return {
        "image_path": "E:/test/sugar_002.jpg",
        "image_filename": "sugar_002.jpg",
        "domain": "sugar",
        "predicted_class": "labile",
        "confidence": 0.88,
        "all_probabilities": {
            "unsaturated": 0.01,
            "metastable": 0.03,
            "intermediate": 0.08,
            "labile": 0.88,
        },
        "inference_time_ms": 75.0,
    }


@pytest.fixture
def sugar_prediction_unsaturated():
    """
    - Sugar prediction
    - Unsaturated class
    """
    return {
        "image_path": "E:/test/sugar_003.jpg",
        "image_filename": "sugar_003.jpg",
        "domain": "sugar",
        "predicted_class": "unsaturated",
        "confidence": 0.95,
        "all_probabilities": {
            "unsaturated": 0.95,
            "metastable": 0.03,
            "intermediate": 0.01,
            "labile": 0.01,
        },
        "inference_time_ms": 70.0,
    }


@pytest.fixture
def sugar_prediction_intermediate():
    """
    Sugar prediction
    Intermediate class
    """
    return {
        "image_path": "E:/test/sugar_004.jpg",
        "image_filename": "sugar_004.jpg",
        "domain": "sugar",
        "predicted_class": "intermediate",
        "confidence": 0.78,
        "all_probabilities": {
            "unsaturated": 0.05,
            "metastable": 0.10,
            "intermediate": 0.78,
            "labile": 0.07,
        },
        "inference_time_ms": 82.0,
    }


@pytest.fixture
def mock_flask_app():
    """
    - Creates a Flask test client
    - Mocks heavy dependencies
    """
    with patch("models.loader.get_model") as mock_model, \
         patch("models.loader.get_device", return_value="cpu"), \
         patch("database.mongo_client.check_connection", return_value=True):
        mock_model.return_value = MagicMock()
        from app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client