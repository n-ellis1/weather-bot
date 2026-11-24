"""Mocked-HTTP tests for the OpenWeather 5-day / 3-hour forecast tool."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

import weather_tools
from weather_tools import get_weather_forecast

FAKE_KEY = "test-secret-key-1234567890abcdef"
CST = -6 * 3600  # Amarillo in November (UTC-06:00)


def _ts(year, month, day, hour, minute=0) -> int:
    return int(datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp())


def _sample(dt: int, temp: float, description: str = "clear sky") -> dict:
    return {
        "dt": dt,
        "main": {"temp": temp, "temp_min": temp, "temp_max": temp},
        "weather": [{"main": "Clear", "description": description}],
    }


def _payload(samples, offset=CST, name="Amarillo", country="US") -> dict:
    return {
        "cod": "200",
        "cnt": len(samples),
        "list": samples,
        "city": {"name": name, "country": country, "timezone": offset},
    }


def _full_window(start_utc: int, count: int = 40) -> list[dict]:
    return [_sample(start_utc + i * 3 * 3600, 10.0 + i) for i in range(count)]


def _response(status: int, payload=None) -> MagicMock:
    resp = MagicMock(status_code=status)
    resp.json.return_value = payload if payload is not None else {}
    return resp


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", FAKE_KEY)


def _invoke_with(payload, **args):
    with patch.object(weather_tools.requests, "get", return_value=_response(200, payload)) as get:
        result = get_weather_forecast.invoke({"city": "Amarillo", **args})
    return result, get


def _assert_no_key(result):
    assert FAKE_KEY not in str(result)


# --- schema -----------------------------------------------------------------


def test_tool_schema_is_structured():
    schema = get_weather_forecast.args
    assert set(schema) == {"city", "units"}
    assert schema["units"]["enum"] == ["metric", "imperial"]
    assert schema["units"]["default"] == "metric"
    assert get_weather_forecast.name == "get_weather_forecast"


def test_invalid_units_rejected_by_schema():
    with pytest.raises(Exception):
        get_weather_forecast.invoke({"city": "Amarillo", "units": "kelvin"})


# --- daily grouping ---------------------------------------------------------


def test_groups_by_local_date_not_utc():
    # 2026-11-24 03:00 UTC is 2026-11-23 21:00 in Amarillo (UTC-6).
    samples = _full_window(_ts(2026, 11, 24, 3))
    result, _ = _invoke_with(_payload(samples))

    assert result["ok"] is True
    assert result["location"] == "Amarillo, US"
    assert result["utc_offset"] == "-06:00"
    days = result["days"]
    assert [d["date"] for d in days] == [
        "2026-11-23", "2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27", "2026-11-28",
    ]
    assert [d["sample_count"] for d in days] == [1, 8, 8, 8, 8, 7]
    assert [d["partial"] for d in days] == [True, False, False, False, False, True]
    assert days[0]["first_sample_local"] == "21:00"
    assert days[1]["first_sample_local"] == "00:00"
    assert days[1]["last_sample_local"] == "21:00"
    assert days[-1]["last_sample_local"] == "18:00"


def test_high_low_from_samples():
    # Temperatures rise by 1 per sample, so day 2 (samples 1..8) spans 11..18.
    samples = _full_window(_ts(2026, 11, 24, 3))
    result, _ = _invoke_with(_payload(samples))
    day = result["days"][1]
    assert (day["low"], day["high"]) == (11.0, 18.0)
    assert (result["days"][0]["low"], result["days"][0]["high"]) == (10.0, 10.0)


def test_uses_sample_temp_not_spatial_min_max():
    sample = _sample(_ts(2026, 11, 24, 18), 15.0)
    sample["main"].update(temp_min=-40.0, temp_max=60.0)
    result, _ = _invoke_with(_payload([sample]))
    assert (result["days"][0]["low"], result["days"][0]["high"]) == (15.0, 15.0)


def test_unsorted_samples_are_ordered():
    a = _sample(_ts(2026, 11, 24, 15), 5.0, "light rain")
    b = _sample(_ts(2026, 11, 24, 18), 7.0, "light rain")
    result, _ = _invoke_with(_payload([b, a]))
    day = result["days"][0]
    assert day["first_sample_local"] == "09:00"
    assert day["last_sample_local"] == "12:00"


def test_dominant_conditions_and_tie_break():
    base = _ts(2026, 11, 24, 6)  # 00:00 local
    step = 3 * 3600
    samples = [
        _sample(base + 0 * step, 1, "few clouds"),
        _sample(base + 1 * step, 2, "light rain"),
        _sample(base + 2 * step, 3, "light rain"),
        _sample(base + 3 * step, 4, "few clouds"),
        _sample(base + 4 * step, 5, "light rain"),
        # next local day: a tie, so the earliest description wins
        _sample(base + 8 * step, 6, "snow"),
        _sample(base + 9 * step, 7, "clear sky"),
    ]
    result, _ = _invoke_with(_payload(samples))
    first, second = result["days"]
    assert first["conditions"] == "light rain"
    assert first["other_conditions"] == ["few clouds"]
    assert second["conditions"] == "snow"
    assert second["other_conditions"] == ["clear sky"]


def test_half_hour_offset():
    # India (UTC+05:30): 2026-11-24 20:00 UTC is 2026-11-25 01:30 local.
    samples = [_sample(_ts(2026, 11, 24, 17), 20.0), _sample(_ts(2026, 11, 24, 20), 18.0)]
    result, _ = _invoke_with(_payload(samples, offset=19800, name="Delhi", country="IN"))
    assert result["utc_offset"] == "+05:30"
    assert [(d["date"], d["first_sample_local"]) for d in result["days"]] == [
        ("2026-11-24", "22:30"),
        ("2026-11-25", "01:30"),
    ]


def test_missing_country_and_weather_entries():
    sample = _sample(_ts(2026, 11, 24, 18), 3.0)
    sample["weather"] = []
    result, _ = _invoke_with(_payload([sample], country=None))
    assert result["location"] == "Amarillo"
    assert result["days"][0]["conditions"] == "unknown"


# --- units ------------------------------------------------------------------


def test_metric_request_and_labels():
    result, get = _invoke_with(_payload(_full_window(_ts(2026, 11, 24, 3))), units="metric")
    assert result["units"] == "metric"
    assert result["unit_labels"] == {"temperature": "°C"}
    assert get.call_args.args[0] == weather_tools.FORECAST_URL
    assert get.call_args.kwargs["params"] == {"q": "Amarillo", "units": "metric", "appid": FAKE_KEY}
    assert get.call_args.kwargs["timeout"] == weather_tools.REQUEST_TIMEOUT_SECONDS


def test_imperial_request_and_labels():
    result, get = _invoke_with(_payload(_full_window(_ts(2026, 11, 24, 3))), units="imperial")
    assert result["units"] == "imperial"
    assert result["unit_labels"] == {"temperature": "°F"}
    assert get.call_args.kwargs["params"]["units"] == "imperial"


def test_default_units_is_metric():
    result, get = _invoke_with(_payload(_full_window(_ts(2026, 11, 24, 3))))
    assert result["units"] == "metric"
    assert get.call_args.kwargs["params"]["units"] == "metric"


def test_success_result_does_not_contain_key():
    result, _ = _invoke_with(_payload(_full_window(_ts(2026, 11, 24, 3))))
    _assert_no_key(result)


# --- errors -----------------------------------------------------------------


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    with patch.object(weather_tools.requests, "get") as get:
        result = get_weather_forecast.invoke({"city": "Amarillo"})
    assert result["ok"] is False
    assert "missing OpenWeather API key" in result["error"]
    get.assert_not_called()


def test_blank_city_skips_request():
    with patch.object(weather_tools.requests, "get") as get:
        result = weather_tools.fetch_weather_forecast("   ")
    assert result["ok"] is False
    get.assert_not_called()


def test_unknown_city():
    with patch.object(weather_tools.requests, "get", return_value=_response(404, {"cod": "404"})):
        result = get_weather_forecast.invoke({"city": "Notarealcityzzz"})
    assert result["ok"] is False
    assert "City not found" in result["error"]
    assert "Notarealcityzzz" in result["error"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "rejected the API credentials"), (429, "rate limit"), (500, "HTTP 500"), (503, "HTTP 503")],
)
def test_http_errors(status, expected):
    with patch.object(weather_tools.requests, "get", return_value=_response(status, {"message": FAKE_KEY})):
        result = get_weather_forecast.invoke({"city": "Amarillo"})
    assert result["ok"] is False
    assert expected in result["error"]
    _assert_no_key(result)


def test_timeout():
    with patch.object(weather_tools.requests, "get", side_effect=requests.Timeout("timed out")):
        result = get_weather_forecast.invoke({"city": "Amarillo"})
    assert result["ok"] is False
    assert "timed out" in result["error"]


def test_connection_error_does_not_leak_key():
    leaky = requests.ConnectionError(f"Max retries exceeded with url: /data/2.5/forecast?q=Amarillo&appid={FAKE_KEY}")
    with patch.object(weather_tools.requests, "get", side_effect=leaky):
        result = get_weather_forecast.invoke({"city": "Amarillo"})
    assert result["ok"] is False
    assert "Could not reach" in result["error"]
    _assert_no_key(result)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"list": [_sample(_ts(2026, 11, 24, 18), 1.0)]},  # no city
        {"city": {"name": "Amarillo", "timezone": CST}},  # no list
        {"city": {"name": "Amarillo", "timezone": CST}, "list": []},
        {"city": {"name": "Amarillo", "timezone": CST}, "list": [{"dt": 1}]},  # sample without main
        {"city": {"name": "Amarillo", "timezone": CST}, "list": [{"main": {"temp": 1}}]},  # sample without dt
        {"city": "Amarillo", "list": [_sample(_ts(2026, 11, 24, 18), 1.0)]},  # city not an object
        [],
    ],
)
def test_malformed_response(payload):
    with patch.object(weather_tools.requests, "get", return_value=_response(200, payload)):
        result = get_weather_forecast.invoke({"city": "Amarillo"})
    assert result == {"ok": False, "error": "The weather service returned an unexpected response."}


def test_non_json_response():
    resp = _response(200)
    resp.json.side_effect = ValueError("not json")
    with patch.object(weather_tools.requests, "get", return_value=resp):
        result = get_weather_forecast.invoke({"city": "Amarillo"})
    assert result["ok"] is False
    assert "unexpected response" in result["error"]
