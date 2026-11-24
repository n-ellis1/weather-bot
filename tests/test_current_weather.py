"""Mocked-HTTP tests for the OpenWeather current weather tool."""

from unittest.mock import MagicMock, patch

import pytest
import requests

import weather_tools
from weather_tools import get_current_weather

FAKE_KEY = "test-secret-key-1234567890abcdef"

AMARILLO_PAYLOAD = {
    "name": "Amarillo",
    "sys": {"country": "US"},
    "main": {"temp": 12.3, "feels_like": 10.1, "humidity": 40},
    "wind": {"speed": 6.2},
    "weather": [{"description": "clear sky"}],
}


def _response(status: int, payload=None) -> MagicMock:
    resp = MagicMock(status_code=status)
    resp.json.return_value = payload if payload is not None else {}
    return resp


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", FAKE_KEY)


def _invoke(**args):
    return get_current_weather.invoke({"city": "Amarillo", **args})


def _assert_no_key(result):
    assert FAKE_KEY not in str(result)


def test_tool_schema_is_structured():
    schema = get_current_weather.args
    assert set(schema) == {"city", "units"}
    assert schema["units"]["enum"] == ["metric", "imperial"]
    assert get_current_weather.name == "get_current_weather"


def test_success_metric():
    with patch.object(weather_tools.requests, "get", return_value=_response(200, AMARILLO_PAYLOAD)) as get:
        result = _invoke(units="metric")

    assert result == {
        "ok": True,
        "location": "Amarillo, US",
        "units": "metric",
        "temperature": 12.3,
        "feels_like": 10.1,
        "humidity_percent": 40,
        "wind_speed": 6.2,
        "description": "clear sky",
        "unit_labels": {"temperature": "°C", "wind_speed": "m/s"},
    }
    kwargs = get.call_args.kwargs
    assert kwargs["params"] == {"q": "Amarillo", "units": "metric", "appid": FAKE_KEY}
    assert kwargs["timeout"] == weather_tools.REQUEST_TIMEOUT_SECONDS


def test_success_imperial_labels_and_default_units():
    with patch.object(weather_tools.requests, "get", return_value=_response(200, AMARILLO_PAYLOAD)):
        imperial = _invoke(units="imperial")
    assert imperial["unit_labels"] == {"temperature": "°F", "wind_speed": "mph"}

    with patch.object(weather_tools.requests, "get", return_value=_response(200, AMARILLO_PAYLOAD)) as get:
        default = _invoke()
    assert default["units"] == "metric"
    assert get.call_args.kwargs["params"]["units"] == "metric"


def test_invalid_units_rejected_by_schema():
    with pytest.raises(Exception):
        _invoke(units="kelvin")


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    with patch.object(weather_tools.requests, "get") as get:
        result = _invoke()
    assert result["ok"] is False
    assert "missing OpenWeather API key" in result["error"]
    get.assert_not_called()


def test_unknown_city():
    payload = {"cod": "404", "message": "city not found"}
    with patch.object(weather_tools.requests, "get", return_value=_response(404, payload)):
        result = get_current_weather.invoke({"city": "Notarealcityzzz"})
    assert result["ok"] is False
    assert "City not found" in result["error"]
    assert "Notarealcityzzz" in result["error"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "rejected the API credentials"), (429, "rate limit"), (500, "HTTP 500"), (503, "HTTP 503")],
)
def test_http_errors(status, expected):
    with patch.object(weather_tools.requests, "get", return_value=_response(status, {"message": FAKE_KEY})):
        result = _invoke()
    assert result["ok"] is False
    assert expected in result["error"]
    _assert_no_key(result)


def test_timeout():
    with patch.object(weather_tools.requests, "get", side_effect=requests.Timeout("timed out")):
        result = _invoke()
    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_connection_error_does_not_leak_key():
    leaky = requests.ConnectionError(f"Max retries exceeded with url: /data/2.5/weather?q=Amarillo&appid={FAKE_KEY}")
    with patch.object(weather_tools.requests, "get", side_effect=leaky):
        result = _invoke()
    assert result["ok"] is False
    assert "Could not reach" in result["error"]
    _assert_no_key(result)


def test_malformed_response():
    with patch.object(weather_tools.requests, "get", return_value=_response(200, {"name": "Amarillo"})):
        result = _invoke()
    assert result == {"ok": False, "error": "The weather service returned an unexpected response."}


def test_non_json_response():
    resp = _response(200)
    resp.json.side_effect = ValueError("not json")
    with patch.object(weather_tools.requests, "get", return_value=resp):
        result = _invoke()
    assert result["ok"] is False
    assert "unexpected response" in result["error"]


def test_blank_city_skips_request():
    with patch.object(weather_tools.requests, "get") as get:
        result = weather_tools.fetch_current_weather("   ")
    assert result["ok"] is False
    get.assert_not_called()
