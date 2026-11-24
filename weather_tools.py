"""OpenWeather-backed LangChain tools for Weather-Bot."""

from __future__ import annotations

import os
from typing import Any, Literal

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT_SECONDS = 10

_UNIT_LABELS = {
    "metric": {"temperature": "°C", "wind_speed": "m/s"},
    "imperial": {"temperature": "°F", "wind_speed": "mph"},
}


class CurrentWeatherInput(BaseModel):
    """Arguments for the current weather tool."""

    city: str = Field(
        description="City name, optionally with state/country code, e.g. 'Amarillo' or 'Paris,FR'.",
        min_length=1,
    )
    units: Literal["metric", "imperial"] = Field(
        default="metric",
        description="'metric' for °C and m/s, 'imperial' for °F and mph.",
    )


def _error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def fetch_current_weather(city: str, units: str = "metric") -> dict[str, Any]:
    """Call OpenWeather's current weather endpoint and return a compact summary.

    Always returns a dict. On success ``ok`` is True; on failure ``ok`` is False
    and ``error`` holds a user-safe message (never the API key or request URL).
    """
    city = (city or "").strip()
    if not city:
        return _error("Please provide a city name.")
    if units not in _UNIT_LABELS:
        return _error("Units must be 'metric' or 'imperial'.")

    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return _error("Weather service is not configured (missing OpenWeather API key).")

    try:
        response = requests.get(
            CURRENT_WEATHER_URL,
            params={"q": city, "units": units, "appid": api_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return _error("The weather service timed out. Please try again shortly.")
    except requests.RequestException:
        # Exception text can include the request URL (and therefore the key); don't surface it.
        return _error("Could not reach the weather service. Please try again later.")

    status = response.status_code
    if status == 404:
        return _error(f"City not found: '{city}'. Check the spelling or add a country code.")
    if status == 401:
        return _error("The weather service rejected the API credentials.")
    if status == 429:
        return _error("The weather service rate limit was reached. Please try again later.")
    if status != 200:
        return _error(f"The weather service returned an error (HTTP {status}).")

    try:
        data = response.json()
        main = data["main"]
        weather = data.get("weather") or [{}]
        location = data["name"]
        country = data.get("sys", {}).get("country")
        summary = {
            "ok": True,
            "location": f"{location}, {country}" if country else location,
            "units": units,
            "temperature": main["temp"],
            "feels_like": main["feels_like"],
            "humidity_percent": main["humidity"],
            "wind_speed": data.get("wind", {}).get("speed"),
            "description": weather[0].get("description", "unknown"),
            "unit_labels": _UNIT_LABELS[units],
        }
    except (ValueError, KeyError, TypeError, IndexError):
        return _error("The weather service returned an unexpected response.")

    return summary


@tool("get_current_weather", args_schema=CurrentWeatherInput)
def get_current_weather(city: str, units: Literal["metric", "imperial"] = "metric") -> dict[str, Any]:
    """Get the current weather for a city: resolved location, temperature, feels-like
    temperature, humidity, wind speed, and a short conditions description."""
    return fetch_current_weather(city, units)
