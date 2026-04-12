"""
API Endpoint Tests
    - Tests Flask routes with
        - mocked model
        - MongoDB
        - Gemini dependencies
"""

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from bson import ObjectId

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def app_client():
    """
    - Creates a Flask test client
    - All heavy dependencies mocked
    """
    with patch("database.mongo_client.MongoClient") as mock_mongo, \
         patch("database.mongo_client.check_connection", return_value=True), \
         patch("models.loader._model", new=MagicMock()), \
         patch("models.loader.device", new="cpu"):

        mock_db = MagicMock()
        mock_mongo.return_value.__getitem__ = MagicMock(return_value=mock_db)

        from app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


class TestHealthEndpoint:
    """
    - Tests for GET /api/health
    """

    def test_health_returns_200(self, app_client):
        with patch("models.loader.get_device", return_value="cpu"), \
             patch("database.mongo_client.check_connection", return_value=True):
            resp = app_client.get("/api/health")
            assert resp.status_code == 200

    def test_health_response_structure(self, app_client):
        with patch("models.loader.get_device", return_value="cpu"), \
             patch("database.mongo_client.check_connection", return_value=True):
            resp = app_client.get("/api/health")
            data = resp.get_json()

            assert data["status"] == "ok"
            assert data["service"] == "CaneNexus Backend"
            assert "device" in data
            assert "mongodb" in data

    def test_health_reports_device(self, app_client):
        with patch("models.loader.get_device", return_value="cuda:0"), \
             patch("database.mongo_client.check_connection", return_value=True):
            resp = app_client.get("/api/health")
            data = resp.get_json()
            assert data["device"] == "cuda:0"

    def test_health_reports_mongodb_status(self, app_client):
        with patch("models.loader.get_device", return_value="cpu"), \
             patch("database.mongo_client.check_connection", return_value=False):
            resp = app_client.get("/api/health")
            data = resp.get_json()
            assert data["mongodb"] is False


class TestPredictEndpoint:
    """
    - Tests for POST /api/predict
    """

    def test_predict_rejects_missing_body(self, app_client):
        resp = app_client.post("/api/predict", content_type="application/json")
        assert resp.status_code == 400

    def test_predict_rejects_missing_image_path(self, app_client):
        resp = app_client.post(
            "/api/predict",
            data=json.dumps({"domain": "steel"}),
            content_type="application/json"
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "image_path" in data["error"].lower() or "required" in data["error"].lower()

    def test_predict_rejects_invalid_domain(self, app_client):
        resp = app_client.post(
            "/api/predict",
            data=json.dumps({"image_path": "E:/test/img.jpg", "domain": "plastic"}),
            content_type="application/json"
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "domain" in data["error"].lower()

    def test_predict_rejects_missing_file(self, app_client):
        resp = app_client.post(
            "/api/predict",
            data=json.dumps({
                "image_path": "E:/nonexistent/image.jpg",
                "domain": "steel"
            }),
            content_type="application/json"
        )
        assert resp.status_code == 404

    @patch("routes.predict.run_pipeline")
    @patch("os.path.isfile", return_value=True)
    def test_predict_success_returns_result(self, mock_isfile, mock_pipeline, app_client):
        mock_pipeline.return_value = {
            "log_id": "abc123",
            "session_id": "sess-1",
            "domain": "steel",
            "prediction": {"domain": "steel"},
            "knowledge_graph": {"decision": "Accept_Strip"},
            "gemini_response": "Analysis complete",
            "step_times": {"inference_ms": 100},
            "total_processing_ms": 200
        }
        resp = app_client.post(
            "/api/predict",
            data=json.dumps({
                "image_path": "E:/test/real_image.jpg",
                "domain": "steel"
            }),
            content_type="application/json"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["domain"] == "steel"
        assert "log_id" in data


class TestLogsEndpoint:
    """
    - Tests for GET /api/logs
    """

    @patch("routes.logs.logs_collection")
    def test_logs_returns_paginated_structure(self, mock_logs, app_client):
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.skip.return_value.limit.return_value = []
        mock_logs.find.return_value = mock_cursor
        mock_logs.count_documents.return_value = 0

        resp = app_client.get("/api/logs")
        assert resp.status_code == 200
        data = resp.get_json()

        assert "logs" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        assert "total_pages" in data

    @patch("routes.logs.logs_collection")
    def test_logs_respects_domain_filter(self, mock_logs, app_client):
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.skip.return_value.limit.return_value = []
        mock_logs.find.return_value = mock_cursor
        mock_logs.count_documents.return_value = 0

        app_client.get("/api/logs?domain=steel")
        call_args = mock_logs.find.call_args[0][0]
        assert call_args.get("domain") == "steel"

    @patch("routes.logs.logs_collection")
    def test_logs_default_page_is_1(self, mock_logs, app_client):
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.skip.return_value.limit.return_value = []
        mock_logs.find.return_value = mock_cursor
        mock_logs.count_documents.return_value = 0

        resp = app_client.get("/api/logs")
        data = resp.get_json()
        assert data["page"] == 1


class TestChatEndpoint:
    """
    - Tests for POST /api/chat
    """

    def test_chat_rejects_missing_body(self, app_client):
        resp = app_client.post("/api/chat", content_type="application/json")
        assert resp.status_code == 400

    def test_chat_rejects_missing_fields(self, app_client):
        resp = app_client.post(
            "/api/chat",
            data=json.dumps({"log_id": "abc123"}),
            content_type="application/json"
        )
        assert resp.status_code == 400

    def test_chat_rejects_empty_message(self, app_client):
        resp = app_client.post(
            "/api/chat",
            data=json.dumps({"log_id": "abc123", "message": "   "}),
            content_type="application/json"
        )
        assert resp.status_code == 400

    @patch("routes.chat.chats_collection")
    @patch("routes.chat.logs_collection")
    @patch("routes.chat.chat_response")
    def test_chat_success_returns_response(
        self, mock_gemini, mock_logs, mock_chats, app_client
    ):
        fake_id = ObjectId()
        mock_logs.find_one.return_value = {
            "_id": fake_id,
            "session_id": "sess-1",
            "model_prediction": {"domain": "steel"},
            "knowledge_graph_output": {"decision": "Accept"}
        }
        mock_chats.find_one.return_value = {
            "log_id": str(fake_id),
            "messages": []
        }
        mock_gemini.return_value = "This steel strip looks good."

        resp = app_client.post(
            "/api/chat",
            data=json.dumps({
                "log_id": str(fake_id),
                "message": "What does this mean?"
            }),
            content_type="application/json"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "response" in data
        assert "history" in data


class TestSimulateEndpoint:
    """
    - Tests for POST /api/simulate
    """

    def test_simulate_rejects_missing_dirs(self, app_client):
        resp = app_client.post(
            "/api/simulate",
            data=json.dumps({"steel_dir": "E:/steel"}),
            content_type="application/json"
        )
        assert resp.status_code == 400

    def test_simulate_rejects_nonexistent_dirs(self, app_client):
        resp = app_client.post(
            "/api/simulate",
            data=json.dumps({
                "steel_dir": "E:/nonexistent_steel",
                "sugar_dir": "E:/nonexistent_sugar"
            }),
            content_type="application/json"
        )
        assert resp.status_code == 404


class TestErrorHandlers:
    """
    - Tests for global error handlers
    """

    def test_404_on_unknown_route(self, app_client):
        resp = app_client.get("/api/unknown_endpoint")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "error" in data