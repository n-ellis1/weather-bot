"""Streamlit AppTest checks: the chat flow is wired in and session history is kept."""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_replies_come_from_run_turn_and_history_persists(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    seen_histories = []

    def fake_run_turn(model, history, prompt):
        seen_histories.append([dict(m) for m in history])
        return {"reply": f"reply to {prompt}", "tool_calls": []}

    with patch("chat_agent.run_turn", side_effect=fake_run_turn), patch("chat_agent.build_model", return_value=object()):
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.chat_input[0].set_value("Weather in Amarillo?").run()
        at.chat_input[0].set_value("And tomorrow?").run()

    assert not at.exception
    assert at.session_state.messages == [
        {"role": "user", "content": "Weather in Amarillo?"},
        {"role": "assistant", "content": "reply to Weather in Amarillo?"},
        {"role": "user", "content": "And tomorrow?"},
        {"role": "assistant", "content": "reply to And tomorrow?"},
    ]
    assert seen_histories[1] == at.session_state.messages[:2]
    assert [m.markdown[0].value for m in at.chat_message][-1] == "reply to And tomorrow?"


def test_missing_xai_key_gives_clear_message(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "")
    with patch("chat_agent.run_turn") as run_turn:
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.chat_input[0].set_value("Weather in Amarillo?").run()
    run_turn.assert_not_called()
    assert "XAI_API_KEY" in at.session_state.messages[-1]["content"]


def test_model_exception_is_reported_without_raw_text(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    err = RuntimeError("POST https://api.x.ai/v1/chat secret-detail")
    with patch("chat_agent.run_turn", side_effect=err), patch("chat_agent.build_model", return_value=object()):
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.chat_input[0].set_value("Weather in Amarillo?").run()
    reply = at.session_state.messages[-1]["content"]
    assert "RuntimeError" in reply and "secret-detail" not in reply and "api.x.ai" not in reply
