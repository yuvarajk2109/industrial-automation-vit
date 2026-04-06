"""
CaneNexus – Gemini Client Tests
Tests chatbot integration with mocked Gemini API.
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGetInitialResponse:
    """Tests for get_initial_response()."""

    @patch("chatbot.gemini_client.genai")
    def test_returns_string_on_success(self, mock_genai):
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This steel strip shows minor defects."

        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response

        from chatbot.gemini_client import get_initial_response

        result = get_initial_response(
            prediction_result={"domain": "steel", "total_defect_area_pct": 0.5},
            kg_result={"decision": "Accept_Strip", "details": "Minor defect"}
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("chatbot.gemini_client.genai")
    def test_passes_domain_in_context(self, mock_genai):
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis complete"

        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response

        from chatbot.gemini_client import get_initial_response

        get_initial_response(
            prediction_result={"domain": "sugar", "predicted_class": "labile"},
            kg_result={"crystal_state": "LABILE"}
        )

        # Verify send_message was called with context containing 'SUGAR'
        call_args = mock_chat.send_message.call_args[0][0]
        assert "SUGAR" in call_args

    @patch("chatbot.gemini_client.genai")
    def test_uses_system_prompt(self, mock_genai):
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"

        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response

        from chatbot.gemini_client import get_initial_response, SYSTEM_PROMPT

        get_initial_response(
            prediction_result={"domain": "steel"},
            kg_result={}
        )

        # Verify GenerativeModel was called with system_instruction
        call_kwargs = mock_genai.GenerativeModel.call_args
        assert call_kwargs[1]["system_instruction"] == SYSTEM_PROMPT

    @patch("chatbot.gemini_client.genai")
    def test_fallback_on_exception(self, mock_genai):
        mock_genai.GenerativeModel.side_effect = Exception("API key invalid")

        from chatbot.gemini_client import get_initial_response

        result = get_initial_response(
            prediction_result={"domain": "steel"},
            kg_result={"details": "Test details here"}
        )

        # Should return fallback string, not raise
        assert isinstance(result, str)
        assert "CaneNexus Analysis Summary" in result
        assert "STEEL" in result

    @patch("chatbot.gemini_client.genai")
    def test_fallback_includes_kg_details(self, mock_genai):
        mock_genai.GenerativeModel.side_effect = RuntimeError("timeout")

        from chatbot.gemini_client import get_initial_response

        result = get_initial_response(
            prediction_result={"domain": "sugar"},
            kg_result={"details": "Crystal state: LABILE, high risk"}
        )

        assert "Crystal state: LABILE" in result


class TestChatResponse:
    """Tests for chat_response()."""

    @patch("chatbot.gemini_client.genai")
    def test_returns_string_on_success(self, mock_genai):
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The defect area is quite small."

        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response

        from chatbot.gemini_client import chat_response

        result = chat_response(
            history=[
                {"role": "model", "content": "Initial assessment here."},
            ],
            user_message="What should I do?",
            prediction_result={"domain": "steel"},
            kg_result={"decision": "Accept_Strip", "details": "Minor"}
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("chatbot.gemini_client.genai")
    def test_converts_history_to_gemini_format(self, mock_genai):
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Reply"

        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response

        from chatbot.gemini_client import chat_response

        history = [
            {"role": "model", "content": "Hello"},
            {"role": "user", "content": "What next?"},
        ]
        chat_response(
            history=history,
            user_message="Follow up",
            prediction_result={"domain": "sugar"},
            kg_result={"details": "test"}
        )

        # Verify start_chat was called with converted history
        call_kwargs = mock_model.start_chat.call_args[1]
        gemini_history = call_kwargs["history"]
        assert len(gemini_history) == 2
        assert gemini_history[0]["role"] == "model"
        assert gemini_history[0]["parts"] == ["Hello"]
        assert gemini_history[1]["role"] == "user"
        assert gemini_history[1]["parts"] == ["What next?"]

    @patch("chatbot.gemini_client.genai")
    def test_includes_context_reminder(self, mock_genai):
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Reply"

        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.start_chat.return_value = mock_chat
        mock_chat.send_message.return_value = mock_response

        from chatbot.gemini_client import chat_response

        chat_response(
            history=[],
            user_message="Question",
            prediction_result={"domain": "steel"},
            kg_result={"details": "Reject strip due to defects"}
        )

        # System instruction should include context reminder
        call_args = mock_genai.GenerativeModel.call_args
        system_instr = call_args[1]["system_instruction"]
        assert "CONTEXT REMINDER" in system_instr
        assert "STEEL" in system_instr

    @patch("chatbot.gemini_client.genai")
    def test_fallback_on_exception(self, mock_genai):
        mock_genai.GenerativeModel.side_effect = Exception("Network error")

        from chatbot.gemini_client import chat_response

        result = chat_response(
            history=[],
            user_message="What happened?",
            prediction_result={"domain": "sugar"},
            kg_result={}
        )

        assert isinstance(result, str)
        assert "apologize" in result.lower() or "trouble" in result.lower()


class TestSystemPrompt:
    """Tests for the SYSTEM_PROMPT constant."""

    def test_system_prompt_exists(self):
        from chatbot.gemini_client import SYSTEM_PROMPT
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 50

    def test_system_prompt_mentions_canenexus(self):
        from chatbot.gemini_client import SYSTEM_PROMPT
        assert "CaneNexus" in SYSTEM_PROMPT

    def test_system_prompt_restricts_knowledge(self):
        from chatbot.gemini_client import SYSTEM_PROMPT
        assert "ONLY" in SYSTEM_PROMPT
        assert "NOT" in SYSTEM_PROMPT

    def test_system_prompt_covers_both_domains(self):
        from chatbot.gemini_client import SYSTEM_PROMPT
        assert "steel" in SYSTEM_PROMPT.lower()
        assert "sugar" in SYSTEM_PROMPT.lower()
