"""OpenWeather-backed LangChain tools for Weather-Bot."""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
REQUEST_TIMEOUT_SECONDS = 10
SAMPLES_PER_FULL_DAY = 8  # 24 h of 3-hour forecast steps
DEFAULT_UNITS = "imperial"  # US units (°F, mph) unless metric is requested

_UNIT_LABELS = {
    "metric": {"temperature": "°C", "wind_speed": "m/s"},
    "imperial": {"temperature": "°F", "wind_speed": "mph"},
}


class CurrentWeatherInput(BaseModel):
    """Arguments for the current weather tool."""

    city: str = Field(
        description=(
            "City name, optionally with country code, e.g. 'Amarillo' or 'Paris,FR'. For a US state "
            "use 'City,ST,US' (e.g. 'Amarillo,TX,US') or the full state name ('Amarillo, Texas')."
        ),
        min_length=1,
    )
    units: Literal["metric", "imperial"] = Field(
        default=DEFAULT_UNITS,
        description="'imperial' for °F and mph (default), 'metric' for °C and m/s.",
    )


class ForecastInput(CurrentWeatherInput):
    """Arguments for the 5-day forecast tool (same as the current weather tool)."""


_UNEXPECTED_RESPONSE = "The weather service returned an unexpected response."


def _error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _get_openweather(url: str, city: str, units: str) -> tuple[Any, dict[str, Any] | None]:
    """Validate inputs, call an OpenWeather endpoint, and decode the JSON body.

    Returns ``(data, None)`` on success or ``(None, error_dict)`` on failure. Error
    messages are user-safe: they never include the API key, request URL, or
    exception text.
    """
    city = (city or "").strip()
    if not city:
        return None, _error("Please provide a city name.")
    if units not in _UNIT_LABELS:
        return None, _error("Units must be 'metric' or 'imperial'.")

    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return None, _error("Weather service is not configured (missing OpenWeather API key).")

    try:
        response = requests.get(
            url,
            params={"q": city, "units": units, "appid": api_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return None, _error("The weather service timed out. Please try again shortly.")
    except requests.RequestException:
        # Exception text can include the request URL (and therefore the key); don't surface it.
        return None, _error("Could not reach the weather service. Please try again later.")

    status = response.status_code
    if status == 404:
        return None, _error(f"City not found: '{city}'. Check the spelling or add a country code.")
    if status == 401:
        return None, _error("The weather service rejected the API credentials.")
    if status == 429:
        return None, _error("The weather service rate limit was reached. Please try again later.")
    if status != 200:
        return None, _error(f"The weather service returned an error (HTTP {status}).")

    try:
        return response.json(), None
    except ValueError:
        return None, _error(_UNEXPECTED_RESPONSE)


def _format_location(name: str, country: str | None) -> str:
    return f"{name}, {country}" if country else name


def fetch_current_weather(city: str, units: str = DEFAULT_UNITS) -> dict[str, Any]:
    """Call OpenWeather's current weather endpoint and return a compact summary.

    Always returns a dict. On success ``ok`` is True; on failure ``ok`` is False
    and ``error`` holds a user-safe message (never the API key or request URL).
    """
    data, error = _get_openweather(CURRENT_WEATHER_URL, city, units)
    if error:
        return error

    try:
        main = data["main"]
        weather = data.get("weather") or [{}]
        location = data["name"]
        country = data.get("sys", {}).get("country")
        offset = data.get("timezone")
        observed = data.get("dt")
        summary = {
            "ok": True,
            "location": _format_location(location, country),
            "units": units,
            "temperature": main["temp"],
            "feels_like": main["feels_like"],
            "humidity_percent": main["humidity"],
            "wind_speed": data.get("wind", {}).get("speed"),
            "pressure_hpa": main.get("pressure"),
            "description": weather[0].get("description", "unknown"),
            "icon": weather[0].get("icon"),
            # Local time of the provider's observation, when both the timestamp and offset are given.
            "observed_local": (
                _local_iso(int(observed), int(offset)) if observed is not None and offset is not None else None
            ),
            "utc_offset": _format_utc_offset(int(offset)) if offset is not None else None,
            "unit_labels": _UNIT_LABELS[units] | {"pressure": "hPa"},
        }
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, OSError):
        return _error(_UNEXPECTED_RESPONSE)

    return summary


def _format_utc_offset(seconds: int) -> str:
    sign = "+" if seconds >= 0 else "-"
    minutes = abs(seconds) // 60
    return f"{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def _local_iso(timestamp: int, offset_seconds: int) -> str:
    """Unix timestamp -> local wall-clock time, e.g. '2026-11-24T13:05'."""
    return datetime.fromtimestamp(timestamp, timezone(timedelta(seconds=offset_seconds))).strftime("%Y-%m-%dT%H:%M")


def _number(value: Any) -> float | None:
    """A finite number from the provider, or ``None`` if it is missing or malformed."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _summarize_forecast_days(samples: list[dict[str, Any]], offset_seconds: int) -> list[dict[str, Any]]:
    """Group 3-hour forecast samples by local calendar date.

    High/low are the max/min of the sampled ``main.temp`` values, so they reflect
    only the sample times, not a true daily extreme. ``conditions`` is the most
    frequent description that day (ties go to the earliest one), and ``icon`` is the
    provider icon of that description's first sample. A day with fewer than 8 samples
    (24 h / 3 h) is marked ``partial``.

    ``precip_chance_percent`` is the highest ``pop`` (probability of precipitation) among
    the day's samples, ``avg_humidity_percent`` the mean sample humidity, and
    ``max_wind_speed`` the highest sample wind speed. Each is ``None`` when no sample
    that day provides the value.
    """
    tz = timezone(timedelta(seconds=offset_seconds))
    days: dict[str, dict[str, Any]] = {}
    for sample in sorted(samples, key=lambda s: s["dt"]):
        local = datetime.fromtimestamp(int(sample["dt"]), tz)
        temp = float(sample["main"]["temp"])
        weather = sample.get("weather") or [{}]
        description = weather[0].get("description") or "unknown"

        day = days.setdefault(
            local.date().isoformat(),
            {"temps": [], "descriptions": [], "times": [], "icons": {}, "pops": [], "humidity": [], "wind": []},
        )
        day["temps"].append(temp)
        day["descriptions"].append(description)
        day["times"].append(local.strftime("%H:%M"))
        if weather[0].get("icon"):
            day["icons"].setdefault(description, weather[0]["icon"])
        if (pop := _number(sample.get("pop"))) is not None:
            day["pops"].append(pop)
        if (humidity := _number(sample["main"].get("humidity"))) is not None:
            day["humidity"].append(humidity)
        if (wind := _number((sample.get("wind") or {}).get("speed"))) is not None:
            day["wind"].append(wind)

    summaries = []
    for date, day in days.items():
        counts = Counter(day["descriptions"])
        # Counter preserves first-insertion order, so max() breaks ties by earliest occurrence.
        dominant = max(counts, key=counts.get)
        summaries.append(
            {
                "date": date,
                "high": max(day["temps"]),
                "low": min(day["temps"]),
                "conditions": dominant,
                "other_conditions": [d for d in counts if d != dominant],
                "icon": day["icons"].get(dominant),
                "precip_chance_percent": round(max(day["pops"]) * 100) if day["pops"] else None,
                "avg_humidity_percent": round(sum(day["humidity"]) / len(day["humidity"])) if day["humidity"] else None,
                "max_wind_speed": max(day["wind"]) if day["wind"] else None,
                "partial": len(day["temps"]) < SAMPLES_PER_FULL_DAY,
                "sample_count": len(day["temps"]),
                "first_sample_local": day["times"][0],
                "last_sample_local": day["times"][-1],
            }
        )
    return summaries


def fetch_weather_forecast(city: str, units: str = DEFAULT_UNITS) -> dict[str, Any]:
    """Call OpenWeather's 5-day / 3-hour forecast endpoint and summarize it per local day.

    Always returns a dict. On success ``ok`` is True and ``days`` lists one entry per
    local date (in the forecast city's time zone offset); on failure ``ok`` is False
    and ``error`` holds a user-safe message (never the API key or request URL).
    """
    data, error = _get_openweather(FORECAST_URL, city, units)
    if error:
        return error

    try:
        city_info = data["city"]
        offset = int(city_info.get("timezone") or 0)
        samples = data["list"]
        if not isinstance(samples, list) or not samples:
            return _error(_UNEXPECTED_RESPONSE)
        days = _summarize_forecast_days(samples, offset)
        summary = {
            "ok": True,
            "location": _format_location(city_info["name"], city_info.get("country")),
            "units": units,
            "utc_offset": _format_utc_offset(offset),
            "days": days,
            "unit_labels": _UNIT_LABELS[units],
            "note": (
                "High/low are the max/min of 3-hourly forecast samples, not true daily extremes. "
                "precip_chance_percent is the highest probability of precipitation among the day's "
                "3-hour samples. Days marked partial have fewer than 8 samples."
            ),
        }
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, OSError):
        return _error(_UNEXPECTED_RESPONSE)

    return summary


@tool("get_current_weather", args_schema=CurrentWeatherInput)
def get_current_weather(city: str, units: Literal["metric", "imperial"] = DEFAULT_UNITS) -> dict[str, Any]:
    """Get the current weather for a city: resolved location, temperature, feels-like
    temperature, humidity, wind speed, pressure, a short conditions description, and the
    local observation time."""
    return fetch_current_weather(city, units)


@tool("get_weather_forecast", args_schema=ForecastInput)
def get_weather_forecast(city: str, units: Literal["metric", "imperial"] = DEFAULT_UNITS) -> dict[str, Any]:
    """Get the 5-day weather forecast for a city, summarized per local date: resolved
    location, and for each day the high, low, dominant weather conditions, precipitation
    chance, average humidity, and maximum wind speed. Days at the start or end of the
    forecast window may be partial (marked ``partial``)."""
    return fetch_weather_forecast(city, units)
