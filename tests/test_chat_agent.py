"""Tests for the Grok chat flow with a scripted fake model and mocked OpenWeather HTTP."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import chat_agent
import weather_tools
from chat_agent import MAX_TOOL_CALLS_PER_TURN, TOOL_LIMIT_MESSAGE, run_turn, to_langchain_history

FAKE_OWM_KEY = "test-owm-secret-1234567890abcdef"

CURRENT_PAYLOAD = {
    "name": "Amarillo",
    "sys": {"country": "US"},
    "main": {"temp": 41.0, "feels_like": 35.2, "humidity": 55},
    "wind": {"speed": 12.5},
    "weather": [{"description": "light snow"}],
}

FORECAST_PAYLOAD = {
    "city": {"name": "Amarillo", "country": "US", "timezone": -21600},
    "list": [
        {"dt": 1795500000 + i * 10800, "main": {"temp": 30.0 + i}, "weather": [{"description": "light snow"}]}
        for i in range(8)
    ],
}


class FakeModel:
    """Duck-typed chat model: returns scripted AIMessages and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []  # (bind kwargs, messages)
        self.bound = []  # (tool names, bind kwargs)

    def bind_tools(self, tools, **kwargs):
        self.bound.append(([t.name for t in tools], kwargs))
        model = self

        class _Bound:
            def invoke(self, messages):
                model.calls.append((kwargs, list(messages)))
                if not model.responses:
                    raise AssertionError("FakeModel ran out of scripted responses")
                return model.responses.pop(0)

        return _Bound()


def _call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _response(status, payload=None):
    resp = MagicMock(status_code=status)
    resp.json.return_value = payload if payload is not None else {}
    return resp


def _tool_messages(model):
    """ToolMessages present in the last request sent to the model."""
    return [m for m in model.calls[-1][1] if isinstance(m, ToolMessage)]


@pytest.fixture(autouse=True)
def owm_key(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", FAKE_OWM_KEY)


def test_only_the_two_weather_tools_are_bound():
    model = FakeModel([AIMessage(content="Hi! Ask me about the weather.")])
    run_turn(model, [], "hello")
    names = {tuple(sorted(n)) for n, _ in model.bound}
    assert names == {("get_current_weather", "get_weather_forecast")}
    assert any(kw.get("tool_choice") == "none" for _, kw in model.bound)


def test_routes_current_weather_question_and_grounds_reply():
    model = FakeModel(
        [
            _call("get_current_weather", {"city": "Amarillo", "units": "imperial"}),
            AIMessage(content="It's 41 °F with light snow in Amarillo, US."),
        ]
    )
    with patch.object(weather_tools.requests, "get", return_value=_response(200, CURRENT_PAYLOAD)) as get:
        result = run_turn(model, [], "What's the weather in Amarillo right now in Fahrenheit?")

    assert get.call_count == 1
    assert get.call_args.args[0] == weather_tools.CURRENT_WEATHER_URL
    assert get.call_args.kwargs["params"]["units"] == "imperial"
    assert result["reply"] == "It's 41 °F with light snow in Amarillo, US."
    assert [c["name"] for c in result["tool_calls"]] == ["get_current_weather"]

    tool_msgs = _tool_messages(model)
    assert len(tool_msgs) == 1 and tool_msgs[0].tool_call_id == "call_1"
    payload = json.loads(tool_msgs[0].content)
    assert payload["ok"] is True and payload["temperature"] == 41.0 and payload["location"] == "Amarillo, US"


def test_routes_forecast_question():
    model = FakeModel(
        [
            _call("get_weather_forecast", {"city": "Amarillo"}),
            AIMessage(content="Forecast for Amarillo, US: light snow, highs near 37."),
        ]
    )
    with patch.object(weather_tools.requests, "get", return_value=_response(200, FORECAST_PAYLOAD)) as get:
        result = run_turn(model, [], "What's the forecast for Amarillo this week?")

    assert get.call_count == 1
    assert get.call_args.args[0] == weather_tools.FORECAST_URL
    assert get.call_args.kwargs["params"]["units"] == "metric"
    assert [c["name"] for c in result["tool_calls"]] == ["get_weather_forecast"]
    payload = json.loads(_tool_messages(model)[0].content)
    assert payload["ok"] is True and payload["days"]
    assert result["reply"].startswith("Forecast for Amarillo")


def test_system_prompt_and_history_are_sent():
    model = FakeModel([AIMessage(content="ok")])
    history = [{"role": "user", "content": "Weather in Paris?"}, {"role": "assistant", "content": "Sunny."}]
    run_turn(model, history, "And tomorrow?")
    sent = model.calls[0][1]
    assert isinstance(sent[0], SystemMessage) and "tool results" in sent[0].content
    assert [type(m) for m in sent[1:]] == [HumanMessage, AIMessage, HumanMessage]
    assert sent[-1].content == "And tomorrow?"


@pytest.mark.parametrize(
    ("tool_name", "status", "expected"),
    [
        ("get_current_weather", 404, "City not found"),
        ("get_weather_forecast", 404, "City not found"),
        ("get_current_weather", 401, "rejected the API credentials"),
        ("get_weather_forecast", 500, "HTTP 500"),
    ],
)
def test_tool_http_errors_reach_the_model(tool_name, status, expected):
    model = FakeModel([_call(tool_name, {"city": "Nowhereville"}), AIMessage(content="That lookup failed.")])
    with patch.object(weather_tools.requests, "get", return_value=_response(status)):
        result = run_turn(model, [], "Weather in Nowhereville?")

    payload = json.loads(_tool_messages(model)[0].content)
    assert payload["ok"] is False and expected in payload["error"]
    assert result["tool_calls"][0]["result"] == payload


def test_connection_error_does_not_leak_key_to_model():
    model = FakeModel([_call("get_current_weather", {"city": "Amarillo"}), AIMessage(content="Failed.")])
    boom = requests.ConnectionError(f"https://api.openweathermap.org/?appid={FAKE_OWM_KEY}")
    with patch.object(weather_tools.requests, "get", side_effect=boom):
        run_turn(model, [], "Weather in Amarillo?")

    sent_text = " ".join(str(m.content) for m in model.calls[-1][1])
    assert "Could not reach the weather service" in sent_text
    assert FAKE_OWM_KEY not in sent_text


def test_empty_final_reply_falls_back_to_tool_error():
    model = FakeModel(
        [_call("get_current_weather", {"city": "Nowhereville"})] * 3 + [AIMessage(content="")]
    )
    with patch.object(weather_tools.requests, "get", return_value=_response(404)):
        result = run_turn(model, [], "Weather in Nowhereville?")
    assert result["reply"].startswith("The weather lookup failed:")
    assert "City not found" in result["reply"]


def test_unknown_tool_is_refused_without_running():
    model = FakeModel([_call("run_shell", {"cmd": "dir"}), AIMessage(content="I can only look up weather.")])
    with patch.object(weather_tools.requests, "get") as get:
        result = run_turn(model, [], "List my files")
    get.assert_not_called()
    payload = json.loads(_tool_messages(model)[0].content)
    assert payload["ok"] is False and "Unknown tool" in payload["error"]
    assert result["reply"] == "I can only look up weather."


def test_invalid_tool_args_return_error():
    model = FakeModel([_call("get_current_weather", {"city": "Amarillo", "units": "kelvin"}), AIMessage(content="x")])
    with patch.object(weather_tools.requests, "get") as get:
        run_turn(model, [], "Weather in kelvin?")
    get.assert_not_called()
    payload = json.loads(_tool_messages(model)[0].content)
    assert payload == {"ok": False, "error": "The tool was called with invalid arguments."}


def test_tool_call_cap_stops_the_loop():
    # A model that keeps asking for 3 tools per round would loop forever without the cap.
    def many_calls(round_no):
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "get_current_weather", "args": {"city": f"City{round_no}{i}"}, "id": f"c{round_no}{i}", "type": "tool_call"}
                for i in range(3)
            ],
        )

    model = FakeModel([many_calls(r) for r in range(3)] + [AIMessage(content="Here's what I found.")])
    with patch.object(weather_tools.requests, "get", return_value=_response(200, CURRENT_PAYLOAD)) as get:
        result = run_turn(model, [], "Weather everywhere?")

    assert get.call_count == MAX_TOOL_CALLS_PER_TURN
    assert len(result["tool_calls"]) == MAX_TOOL_CALLS_PER_TURN
    assert len(model.calls) == 4  # 3 tool rounds + 1 forced final answer
    assert model.calls[-1][0].get("tool_choice") == "none"
    limited = [m for m in _tool_messages(model) if TOOL_LIMIT_MESSAGE in m.content]
    assert len(limited) == 9 - MAX_TOOL_CALLS_PER_TURN
    assert result["reply"] == "Here's what I found."


def test_history_conversion_skips_unknown_roles():
    msgs = to_langchain_history(
        [{"role": "user", "content": "a"}, {"role": "system", "content": "x"}, {"role": "assistant", "content": "b"}]
    )
    assert [type(m) for m in msgs] == [HumanMessage, AIMessage]


def test_build_model_uses_grok_4_7(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    model = chat_agent.build_model()
    assert model.model_name == "grok-4.7"
