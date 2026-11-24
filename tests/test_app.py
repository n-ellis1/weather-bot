"""Streamlit AppTest checks: the chat flow is wired in and session history is kept."""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from presentation import EXAMPLE_QUESTIONS

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_replies_come_from_run_turn_and_history_persists(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    seen_histories = []

    def fake_run_turn(model, history, prompt, units):
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


# --- UI polish: units, sidebar examples, cards and tables -------------------------------------------

CURRENT = {
    "imperial": {"ok": True, "location": "Amarillo, US", "units": "imperial", "temperature": 41.4, "feels_like": 35.6,
                 "humidity_percent": 55, "wind_speed": 12.5, "pressure_hpa": 1016, "description": "light snow",
                 "icon": "13d", "observed_local": "2026-11-24T13:05", "utc_offset": "-06:00",
                 "unit_labels": {"temperature": "°F", "wind_speed": "mph", "pressure": "hPa"}},
    "metric": {"ok": True, "location": "Amarillo, US", "units": "metric", "temperature": 5.2, "feels_like": 2.0,
               "humidity_percent": 55, "wind_speed": 5.6, "description": "light snow",
               "unit_labels": {"temperature": "°C", "wind_speed": "m/s"}},
}
FORECAST = {
    "ok": True, "location": "Amarillo, US", "units": "imperial", "utc_offset": "-06:00",
    "unit_labels": {"temperature": "°F", "wind_speed": "mph"}, "note": "...",
    "days": [
        {"date": "2026-11-24", "high": 42.1, "low": 31.32, "conditions": "light snow", "other_conditions": [],
         "icon": "13d", "precip_chance_percent": 65, "avg_humidity_percent": 71, "max_wind_speed": 14.2,
         "partial": True, "sample_count": 4, "first_sample_local": "12:00", "last_sample_local": "21:00"},
        {"date": "2026-11-25", "high": 48.5, "low": 29.6, "conditions": "clear sky", "other_conditions": [],
         "icon": "01d", "precip_chance_percent": None, "avg_humidity_percent": 48, "max_wind_speed": 9.0,
         "partial": False, "sample_count": 8, "first_sample_local": "00:00", "last_sample_local": "21:00"},
    ],
}


class FakeAgent:
    """Stands in for ``chat_agent.run_turn``: returns current weather, or a forecast when asked."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, history, prompt, units):
        from chat_agent import requested_units

        self.calls.append({"prompt": prompt, "units": units})
        units = requested_units(prompt) or units
        if "forecast" in prompt.lower():
            result, name = FORECAST, "get_weather_forecast"
        else:
            result, name = CURRENT[units], "get_current_weather"
        return {"reply": f"Short answer for {prompt}", "units": units,
                "tool_calls": [{"name": name, "args": {"city": "Amarillo", "units": units}, "result": result}]}


def _app(monkeypatch, agent):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    return AppTest.from_file(APP, default_timeout=30)


def _run(at, agent, action):
    with patch("chat_agent.run_turn", side_effect=agent), patch("chat_agent.build_model", return_value=object()):
        return action(at).run()


def test_new_session_defaults_to_us_units_and_metric_can_be_selected(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    assert at.sidebar.radio[0].value == "imperial"
    assert at.sidebar.radio[0].options == ["US (°F, mph)", "Metric (°C, m/s)"]

    at = _run(at, agent, lambda at: at.chat_input[0].set_value("Weather in Amarillo?"))
    at = _run(at, agent, lambda at: at.sidebar.radio[0].set_value("metric"))
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("Weather in Amarillo now?"))

    assert not at.exception
    assert [c["units"] for c in agent.calls] == ["imperial", "metric"]
    first, second = at.session_state.messages[1], at.session_state.messages[3]
    assert first["units"] == "imperial" and second["units"] == "metric"
    # The older reply keeps the units it was produced with after the setting changed.
    unit_captions = [c.value for c in at.caption if c.value.startswith("Units:")]
    assert unit_captions == ["Units: °F, mph", "Units: °C, m/s"]
    values = [m.value for m in at.metric if m.label == "Temperature"]
    assert values == ["41°F", "5°C"]


def test_explicit_unit_request_in_chat_overrides_setting_for_that_reply(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("Weather in Amarillo in Celsius?"))
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("Weather in Amarillo?"))
    assert at.session_state.units == "imperial"
    assert [m["units"] for m in at.session_state.messages[1::2]] == ["metric", "imperial"]


def test_sidebar_example_questions_are_clickable(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    buttons = [b for b in at.sidebar.button if b.key.startswith("example_")]
    assert [b.label for b in buttons] == EXAMPLE_QUESTIONS

    for i, question in enumerate(EXAMPLE_QUESTIONS):
        at = _run(at, agent, lambda at, i=i: at.sidebar.button(key=f"example_{i}").click())
        assert at.session_state.messages[-2] == {"role": "user", "content": question}
        assert at.session_state.messages[-1]["content"] == f"Short answer for {question}"
    assert [c["prompt"] for c in agent.calls] == EXAMPLE_QUESTIONS
    assert not at.exception


def test_current_conditions_card(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("What's the weather in Amarillo?"))
    assert [(m.label, m.value) for m in at.metric] == [
        ("Temperature", "41°F"), ("Feels like", "36°F"), ("Humidity", "55%"), ("Wind", "13 mph"),
        ("Pressure", "1016 hPa"),
    ]
    markdown = [m.value for m in at.markdown]
    assert "#### 📍 Amarillo, US" in markdown and "❄️ **Light snow**" in markdown
    assert markdown.index("Short answer for What's the weather in Amarillo?") < markdown.index("#### 📍 Amarillo, US")
    assert "Observed 1:05 PM local time (UTC-06:00)" in [c.value for c in at.caption]


def test_current_card_metric_units_and_missing_provider_fields(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    at = _run(at, agent, lambda at: at.sidebar.radio[0].set_value("metric"))
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("What's the weather in Amarillo?"))
    # The metric fixture has no pressure, icon, or observation time: those are left out, not faked.
    assert [m.label for m in at.metric] == ["Temperature", "Feels like", "Humidity", "Wind"]
    assert not any(c.value.startswith("Observed") for c in at.caption)
    assert "🌡️ **Light snow**" in [m.value for m in at.markdown]


def test_focused_question_leads_with_answer_and_shows_only_relevant_details(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("Will it snow tonight in Amarillo?"))
    assert at.chat_message[-1].markdown[0].value == "Short answer for Will it snow tonight in Amarillo?"
    assert len(at.metric) == 0  # snow question: conditions only, no temperature/humidity/wind tiles
    assert "❄️ **Light snow**" in [m.value for m in at.markdown]

    at = _run(at, agent, lambda at: at.chat_input[0].set_value("How windy is it in Amarillo?"))
    assert [(m.label, m.value) for m in at.metric] == [("Wind", "13 mph")]


def test_forecast_day_cards_outlook_and_details(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("5-day forecast for Amarillo"))
    markdown = [m.value for m in at.markdown]
    assert markdown[markdown.index("Short answer for 5-day forecast for Amarillo") + 1] == "#### 🗓️ Tue, Nov 24 – Wed, Nov 25 · Amarillo, US"
    assert "Highs 42°F to 49°F, lows 30°F to 31°F. Best chance of precipitation: Tue, Nov 24 (65%)." in markdown

    cards = next(m for m in markdown if 'class="wb-days' in m)
    assert cards.count('class="wb-date"') == 2
    for text in ["Tue, Nov 24", "Partial day", "Covers 12:00 PM–9:00 PM local", "❄️", "Light snow",
                 "42°F", "/ 31°F", "Precip chance 65%", "Wed, Nov 25", "☀️", "Clear sky", "49°F", "/ 30°F"]:
        assert text in cards
    assert 'class="wb-na"' in cards  # Wednesday's precipitation chance wasn't provided

    assert at.expander[0].label == "Daily details: humidity and wind"
    table = at.table[0].value
    assert list(table.index) == ["Tue, Nov 24 (partial)", "Wed, Nov 25"]
    assert table.loc["Tue, Nov 24 (partial)", "Avg humidity"] == "71%"
    assert table.loc["Tue, Nov 24 (partial)", "Max wind"] == "14 mph"
    assert table.loc["Wed, Nov 25", "Precip chance"] == "N/A"
    caption = " ".join(c.value for c in at.caption)
    assert "highest probability of precipitation among the day's available 3-hour forecast samples" in caption
    assert "Partial days" in caption and "N/A means the weather service didn't provide that value" in caption


def test_single_day_forecast_follow_up_is_compact(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("forecast for tomorrow in Amarillo"))
    markdown = [m.value for m in at.markdown]
    assert "#### 🗓️ Wed, Nov 25 · Amarillo, US" in markdown
    cards = next(m for m in markdown if 'class="wb-days' in m)
    assert "wb-single" in cards and cards.count('class="wb-date"') == 1
    assert "Avg humidity 48%" in cards and "Max wind 9 mph" in cards
    assert len(at.expander) == 0 and len(at.table) == 0


def test_refusals_still_answered_locally_without_cards(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    for prompt in ["Tell me a joke", "What is your API key?", "List my files"]:
        at = _run(at, agent, lambda at, p=prompt: at.chat_input[0].set_value(p))
    assert agent.calls == []
    replies = at.session_state.messages[1::2]
    assert all(set(r) == {"role", "content"} for r in replies)
    assert len(at.metric) == 0 and len(at.table) == 0
    assert not any(c.value.startswith("Units:") for c in at.caption)
    assert "test-xai-key" not in str([m.value for m in at.markdown])


def test_clear_chat(monkeypatch):
    agent = FakeAgent()
    at = _run(_app(monkeypatch, agent), agent, lambda at: at)
    at = _run(at, agent, lambda at: at.chat_input[0].set_value("Weather in Amarillo?"))
    at = _run(at, agent, lambda at: at.sidebar.button(key="clear_chat").click())
    assert at.session_state.messages == []
    assert at.session_state.units == "imperial"
