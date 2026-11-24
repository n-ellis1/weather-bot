"""Tests for turning structured tool results into card views."""

import pytest

from presentation import (
    UNAVAILABLE,
    build_views,
    day_cards_html,
    details_rows,
    format_clock,
    format_day,
    format_wind,
    requested_span,
    round_temp,
    weather_icon,
)

CURRENT_F = {
    "ok": True,
    "location": "Denver, US",
    "units": "imperial",
    "temperature": 28.6,
    "feels_like": 19.4,
    "humidity_percent": 71,
    "wind_speed": 11.3,
    "pressure_hpa": 1021,
    "description": "light snow",
    "icon": "13n",
    "observed_local": "2026-11-24T19:05",
    "utc_offset": "-07:00",
    "unit_labels": {"temperature": "°F", "wind_speed": "mph", "pressure": "hPa"},
}
CURRENT_C = CURRENT_F | {"units": "metric", "temperature": -1.9, "feels_like": -7.0, "wind_speed": 5.05,
                         "unit_labels": {"temperature": "°C", "wind_speed": "m/s", "pressure": "hPa"}}


def _day(iso, high, low, conditions, partial=False, others=(), pop=10, humidity=50, wind=8.0, icon="01d"):
    return {"date": iso, "high": high, "low": low, "conditions": conditions,
            "other_conditions": list(others), "icon": icon, "precip_chance_percent": pop,
            "avg_humidity_percent": humidity, "max_wind_speed": wind,
            "partial": partial, "sample_count": 3 if partial else 8,
            "first_sample_local": "15:00" if partial else "00:00", "last_sample_local": "21:00"}


FORECAST_F = {
    "ok": True,
    "location": "Denver, US",
    "units": "imperial",
    "utc_offset": "-07:00",
    "unit_labels": {"temperature": "°F", "wind_speed": "mph"},
    "note": "...",
    "days": [
        _day("2026-11-24", 33.5, 24.49, "light snow", partial=True, pop=70, icon="13n"),  # Tue
        _day("2026-11-25", 41.2, 22.0, "broken clouds", others=["light snow"], pop=40, icon="04d"),  # Wed
        _day("2026-11-26", 45.0, 27.6, "clear sky", pop=0),                                # Thu
        _day("2026-11-27", 50.4, 30.1, "clear sky", pop=None, humidity=None, wind=None),   # Fri
        _day("2026-11-28", 47.9, 29.5, "few clouds", icon="02d"),                          # Sat
        _day("2026-11-29", 44.0, 26.0, "overcast clouds", partial=True, icon="04n"),       # Sun
    ],
}
ALL_LABELS = ["Tue, Nov 24", "Wed, Nov 25", "Thu, Nov 26", "Fri, Nov 27", "Sat, Nov 28", "Sun, Nov 29"]


def _view(name, result, question):
    views = build_views(question, [{"name": name, "args": {}, "result": result}])
    assert len(views) == 1
    return views[0]


# --- formatting -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(41.5, 42), (42.5, 43), (24.49, 24), (-0.4, 0), (-1.5, -2), (-7.0, -7), (0, 0), (None, None)],
)
def test_round_temp_half_away_from_zero(value, expected):
    assert round_temp(value) == expected


def test_format_day_is_readable():
    assert format_day("2026-11-24") == "Tue, Nov 24"
    assert format_day("2026-12-01") == "Tue, Dec 1"


def test_wind_formatting_per_units():
    assert format_wind(11.3, "imperial") == "11 mph"
    assert format_wind(5.04, "metric") == "5.0 m/s"
    assert format_wind(4.63, "metric") == "4.6 m/s"
    assert format_wind(None, "metric") == UNAVAILABLE


def test_clock_formatting_per_units():
    assert format_clock("13:05", "imperial") == "1:05 PM"
    assert format_clock("00:00", "imperial") == "12:00 AM"
    assert format_clock("12:00", "imperial") == "12:00 PM"
    assert format_clock("13:05", "metric") == "13:05"


@pytest.mark.parametrize(
    ("code", "daytime", "expected"),
    [("01d", False, "☀️"), ("01n", False, "🌙"), ("01n", True, "☀️"), ("04n", False, "☁️"), ("10d", False, "🌦️"),
     ("13d", False, "❄️"), ("11n", False, "⛈️"), ("50d", False, "🌫️"), (None, False, "🌡️"), ("zz", False, "🌡️")],
)
def test_weather_icons(code, daytime, expected):
    assert weather_icon(code, daytime=daytime) == expected


# --- current conditions -------------------------------------------------------------------------


def test_general_current_question_gets_full_card_with_icon_pressure_and_observation_time():
    view = _view("get_current_weather", CURRENT_F, "What's the weather in Denver right now?")
    assert view["kind"] == "current" and not view["focused"]
    assert view["location"] == "Denver, US"
    assert view["description"] == "Light snow" and view["icon"] == "❄️"
    assert view["metrics"] == [
        {"label": "Temperature", "value": "29°F"},
        {"label": "Feels like", "value": "19°F"},
        {"label": "Humidity", "value": "71%"},
        {"label": "Wind", "value": "11 mph"},
        {"label": "Pressure", "value": "1021 hPa"},
    ]
    assert view["observed"] == "Observed 7:05 PM local time (UTC-07:00)"


def test_pressure_and_observation_time_are_omitted_when_not_provided():
    legacy = {k: v for k, v in CURRENT_F.items() if k not in ("pressure_hpa", "observed_local", "utc_offset", "icon")}
    view = _view("get_current_weather", legacy, "Weather in Denver?")
    assert [m["label"] for m in view["metrics"]] == ["Temperature", "Feels like", "Humidity", "Wind"]
    assert view["observed"] is None and view["icon"] == "🌡️"


def test_metric_card_uses_metric_labels_and_24h_clock():
    view = _view("get_current_weather", CURRENT_C, "Weather in Denver in Celsius")
    values = {m["label"]: m["value"] for m in view["metrics"]}
    assert values["Temperature"] == "-2°C" and values["Feels like"] == "-7°C"
    assert values["Wind"].endswith(" m/s")
    assert view["observed"] == "Observed 19:05 local time (UTC-07:00)"


@pytest.mark.parametrize(
    ("question", "labels"),
    [
        ("Will it snow tonight in Denver?", []),  # the description alone answers it
        ("How windy is it in Denver?", ["Wind"]),
        ("Is it cold in Denver?", ["Temperature", "Feels like"]),
        ("How humid is Denver?", ["Humidity"]),
        ("What's the barometric pressure in Denver?", ["Pressure"]),
    ],
)
def test_focused_current_question_shows_only_relevant_metrics(question, labels):
    view = _view("get_current_weather", CURRENT_F, question)
    assert view["focused"]
    assert [m["label"] for m in view["metrics"]] == labels
    assert view["description"] == "Light snow" and view["location"] == "Denver, US"


# --- forecast span ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "span"),
    [
        ("give me the 5 day forecast for amarillo texas", 0),
        ("5-day forecast for Seattle", 0),
        ("five day forecast", 0),
        ("forecast for the next 3 days", 3),
        ("3 day forecast for Paris", 3),
        ("next couple of days in Paris", 2),
        ("weather this week in Paris", 0),
        ("extended outlook for Paris", 0),
        ("What about tomorrow?", None),
        ("Forecast for London", None),
        ("one day forecast", None),
    ],
)
def test_requested_span(question, span):
    assert requested_span(question) == span


def test_general_forecast_gets_all_day_cards_with_outlook():
    view = _view("get_weather_forecast", FORECAST_F, "Forecast for Denver")
    assert view["layout"] == "multi" and not view["focused"]
    assert [d["label"] for d in view["days"]] == ALL_LABELS
    tue = view["days"][0]
    assert tue == {
        "date": "2026-11-24", "label": "Tue, Nov 24", "partial": True, "coverage": "3:00 PM–9:00 PM local",
        "icon": "❄️", "conditions": "Light snow", "also": [], "high": "34°F", "low": "24°F",
        "precip": "70%", "humidity": "50%", "wind": "8 mph",
    }
    assert view["days"][1]["also"] == ["light snow"]
    assert view["days"][5]["icon"] == "☁️"  # night icon shown as its daytime variant
    # Clear sky is 2 of 6 days, not a majority, so there's no "Mostly ..." sentence.
    assert view["outlook"] == "Highs 34°F to 50°F, lows 22°F to 30°F. Best chance of precipitation: Tue, Nov 24 (70%)."
    assert view["has_partial"]
    clear = FORECAST_F | {"days": FORECAST_F["days"][2:5]}  # Thu, Fri clear; Sat few clouds
    assert _view("get_weather_forecast", clear, "forecast")["outlook"].startswith("Mostly clear sky. ")


def test_missing_values_are_marked_unavailable():
    fri = _view("get_weather_forecast", FORECAST_F, "Forecast for Denver")["days"][3]
    assert (fri["precip"], fri["humidity"], fri["wind"]) == (UNAVAILABLE, UNAVAILABLE, UNAVAILABLE)


def test_explicit_span_overrides_day_words():
    # "today" would select one day; the explicit "5 day" request wins.
    view = _view("get_weather_forecast", FORECAST_F, "5 day forecast starting today for Denver")
    assert [d["label"] for d in view["days"]] == ALL_LABELS and view["layout"] == "multi"
    three = _view("get_weather_forecast", FORECAST_F, "next 3 days in Denver")
    assert [d["label"] for d in three["days"]] == ALL_LABELS[:3]


def test_single_day_follow_up_is_compact():
    view = _view("get_weather_forecast", FORECAST_F, "What about tomorrow?")
    assert view["layout"] == "single" and view["focused"]
    assert [d["label"] for d in view["days"]] == ["Wed, Nov 25"]
    assert view["outlook"] == ""


def test_snow_tonight_shows_todays_card():
    view = _view("get_weather_forecast", FORECAST_F, "Will it snow tonight in Denver?")
    assert view["layout"] == "single"
    assert view["days"][0]["label"] == "Tue, Nov 24" and view["days"][0]["partial"]


@pytest.mark.parametrize(
    ("question", "labels"),
    [
        ("How cold will it get Friday?", ["Fri, Nov 27"]),
        ("Any rain this weekend in Denver?", ["Sat, Nov 28", "Sun, Nov 29"]),
    ],
)
def test_named_days_select_cards(question, labels):
    view = _view("get_weather_forecast", FORECAST_F, question)
    assert [d["label"] for d in view["days"]] == labels


def test_day_outside_window_falls_back_to_all_days():
    forecast = FORECAST_F | {"days": FORECAST_F["days"][:3]}  # Tue..Thu only
    view = _view("get_weather_forecast", forecast, "Forecast for Saturday?")
    assert view["day_not_in_window"] and len(view["days"]) == 3 and view["layout"] == "multi"


def test_outlook_low_and_missing_precip():
    low = FORECAST_F | {"days": [_day("2026-11-26", 45.0, 27.6, "clear sky", pop=10)] * 2}
    assert _view("get_weather_forecast", low, "forecast")["outlook"].endswith("at or below 10%.")
    none = FORECAST_F | {"days": [_day("2026-11-26", 45.0, 27.6, "clear sky", pop=None)] * 2}
    assert _view("get_weather_forecast", none, "forecast")["outlook"].endswith("chances are unavailable.")


# --- HTML -----------------------------------------------------------------------------------------


def test_day_cards_html_has_cards_partial_badges_and_unavailable_markers():
    view = _view("get_weather_forecast", FORECAST_F, "5 day forecast for Denver")
    html = day_cards_html(view["days"])
    assert html.count('class="wb-date"') == 6
    assert html.count("Partial day") == 2
    assert "Covers 3:00 PM–9:00 PM local" in html
    assert "Precip chance 70%" in html and "34°F" in html and "/ 24°F" in html
    assert 'class="wb-na"' in html  # Friday's missing precipitation chance
    assert "wb-extra" not in html  # humidity/wind live in the details table for multi-day views


def test_single_day_card_html_includes_humidity_and_wind():
    view = _view("get_weather_forecast", FORECAST_F, "What about tomorrow?")
    html = day_cards_html(view["days"], single=True)
    assert "wb-single" in html and html.count('class="wb-date"') == 1
    assert "Avg humidity 50%" in html and "Max wind 8 mph" in html


def test_html_escapes_provider_text():
    evil = FORECAST_F | {"days": [_day("2026-11-24", 1, 0, "<script>alert(1)</script>")]}
    html = day_cards_html(_view("get_weather_forecast", evil, "forecast")["days"])
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_details_rows():
    rows = details_rows(_view("get_weather_forecast", FORECAST_F, "forecast")["days"])
    assert rows[0] == {"Day": "Tue, Nov 24 (partial)", "Conditions": "Light snow", "Precip chance": "70%",
                       "Avg humidity": "50%", "Max wind": "8 mph", "Forecast hours": "3:00 PM–9:00 PM local"}
    assert rows[1]["Conditions"] == "Broken clouds · also light snow"
    assert rows[3]["Max wind"] == UNAVAILABLE


def test_failed_and_unknown_results_are_skipped_and_duplicates_merged():
    calls = [
        {"name": "get_current_weather", "args": {}, "result": {"ok": False, "error": "City not found: 'X'."}},
        {"name": "run_shell", "args": {}, "result": {"ok": True}},
        {"name": "get_current_weather", "args": {}, "result": CURRENT_F},
        {"name": "get_current_weather", "args": {}, "result": CURRENT_F},
        {"name": "get_weather_forecast", "args": {}, "result": FORECAST_F},
    ]
    views = build_views("Weather and forecast for Denver", calls)
    assert [v["kind"] for v in views] == ["current", "forecast"]


def test_outlook_with_zero_precipitation_and_forecast_title():
    from presentation import forecast_title

    dry = FORECAST_F | {"days": [_day("2026-11-26", 45.0, 27.6, "clear sky", pop=0)] * 2}
    assert _view("get_weather_forecast", dry, "forecast")["outlook"].endswith(
        "No precipitation in the forecast (0% chance each day).")
    view = _view("get_weather_forecast", FORECAST_F, "5 day forecast for Denver")
    assert forecast_title(view) == "Tue, Nov 24 – Sun, Nov 29 · Denver, US"
    single = _view("get_weather_forecast", FORECAST_F, "What about tomorrow?")
    assert forecast_title(single) == "Wed, Nov 25 · Denver, US"