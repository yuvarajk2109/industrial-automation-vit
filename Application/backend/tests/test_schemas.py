"""
CaneNexus – Schema Tests
Validates MongoDB document factory functions produce correct structures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime
from database.schemas import (
    create_log_document,
    create_simulation_document,
    create_chat_document
)


class TestLogDocument:
    """Tests for create_log_document()."""

    def test_returns_dict(self):
        doc = create_log_document(
            session_id="test-session",
            image_path="E:/test/img.jpg",
            image_filename="img.jpg",
            domain="steel",
            model_prediction={"domain": "steel"},
            knowledge_graph_output={"decision": "Accept"},
            gemini_initial_response="Test response",
            processing_time_ms=150.0,
            device="cpu"
        )
        assert isinstance(doc, dict)

    def test_has_required_fields(self):
        doc = create_log_document(
            session_id="s-123",
            image_path="/path/to/img.jpg",
            image_filename="img.jpg",
            domain="sugar",
            model_prediction={},
            knowledge_graph_output={},
            gemini_initial_response="response",
            processing_time_ms=100.0,
            device="cuda"
        )
        required = [
            "session_id", "timestamp", "image_path", "image_filename",
            "domain", "model_prediction", "knowledge_graph_output",
            "gemini_initial_response", "processing_time_ms", "metadata"
        ]
        for key in required:
            assert key in doc, f"Missing key: {key}"

    def test_timestamp_is_datetime(self):
        doc = create_log_document(
            session_id="s-1", image_path="p", image_filename="f",
            domain="steel", model_prediction={},
            knowledge_graph_output={},
            gemini_initial_response="r",
            processing_time_ms=50.0, device="cpu"
        )
        assert isinstance(doc["timestamp"], datetime)

    def test_metadata_has_model_info(self):
        doc = create_log_document(
            session_id="s-2", image_path="p", image_filename="f",
            domain="steel", model_prediction={},
            knowledge_graph_output={},
            gemini_initial_response="r",
            processing_time_ms=50.0, device="cuda"
        )
        meta = doc["metadata"]
        assert meta["model_name"] == "DDA-ViT"
        assert meta["steel_backbone"] == "mit_b4"
        assert meta["sugar_backbone"] == "swin_tiny_patch4_window7_224"
        assert meta["device"] == "cuda"

    def test_preserves_domain(self):
        for domain in ["steel", "sugar"]:
            doc = create_log_document(
                session_id="s", image_path="p", image_filename="f",
                domain=domain, model_prediction={},
                knowledge_graph_output={},
                gemini_initial_response="r",
                processing_time_ms=1.0, device="cpu"
            )
            assert doc["domain"] == domain


class TestSimulationDocument:
    """Tests for create_simulation_document()."""

    def test_returns_dict(self):
        doc = create_simulation_document(
            session_id="sim-001",
            steel_directory="E:/steel",
            sugar_directory="E:/sugar",
            total_steel_images=100,
            total_sugar_images=80
        )
        assert isinstance(doc, dict)

    def test_initial_status_is_running(self):
        doc = create_simulation_document(
            session_id="sim-001",
            steel_directory="E:/steel",
            sugar_directory="E:/sugar",
            total_steel_images=50,
            total_sugar_images=40
        )
        assert doc["status"] == "running"

    def test_completed_at_is_none_initially(self):
        doc = create_simulation_document(
            session_id="sim-001",
            steel_directory="E:/steel",
            sugar_directory="E:/sugar",
            total_steel_images=10,
            total_sugar_images=10
        )
        assert doc["completed_at"] is None

    def test_summary_has_correct_structure(self):
        doc = create_simulation_document(
            session_id="sim-001",
            steel_directory="E:/steel",
            sugar_directory="E:/sugar",
            total_steel_images=10,
            total_sugar_images=10
        )
        steel_summary = doc["summary"]["steel"]
        assert steel_summary["accept"] == 0
        assert steel_summary["downgrade"] == 0
        assert steel_summary["reject"] == 0
        assert steel_summary["manual_inspection"] == 0

        sugar_summary = doc["summary"]["sugar"]
        assert sugar_summary["unsaturated"] == 0
        assert sugar_summary["metastable"] == 0
        assert sugar_summary["intermediate"] == 0
        assert sugar_summary["labile"] == 0

    def test_processed_count_starts_at_zero(self):
        doc = create_simulation_document(
            session_id="sim-001",
            steel_directory="E:/steel",
            sugar_directory="E:/sugar",
            total_steel_images=10,
            total_sugar_images=10
        )
        assert doc["processed_count"] == 0


class TestChatDocument:
    """Tests for create_chat_document()."""

    def test_returns_dict(self):
        doc = create_chat_document(
            log_id="abc123",
            session_id="session-1",
            initial_message="Hello, here is your analysis."
        )
        assert isinstance(doc, dict)

    def test_has_initial_message(self):
        msg = "Test initial message"
        doc = create_chat_document(
            log_id="abc123",
            session_id="session-1",
            initial_message=msg
        )
        assert len(doc["messages"]) == 1
        assert doc["messages"][0]["role"] == "model"
        assert doc["messages"][0]["content"] == msg

    def test_initial_message_has_timestamp(self):
        doc = create_chat_document(
            log_id="abc123",
            session_id="session-1",
            initial_message="test"
        )
        assert isinstance(doc["messages"][0]["timestamp"], datetime)

    def test_references_log_id(self):
        doc = create_chat_document(
            log_id="log_xyz",
            session_id="session-1",
            initial_message="test"
        )
        assert doc["log_id"] == "log_xyz"
        assert doc["session_id"] == "session-1"
