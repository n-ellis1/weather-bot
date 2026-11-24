"""Turn structured weather tool results into display-ready views for the Streamlit UI.

Views are plain, JSON-friendly dicts built once when a reply is produced and stored with
that reply, so older replies keep the values and units they were originally shown with.
Every displayed measurement comes from the tool result of that turn; nothing is taken from
the model's text.

* ``current`` views back the current-conditions card.
* ``forecast`` views back the forecast day cards (``layout`` ``"multi"`` for a span of days,
  ``"single"`` for a one-day follow-up such as "What about tomorrow?").

For a focused current-weather question ("How windy is it?") only the relevant metrics are
kept. For forecasts, an explicit span ("5 day forecast", "next 3 days", "this week") wins over
day words; otherwise named days ("tomorrow", "Friday", "weekend") select the days shown.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime, timedelta
from html import escape
from typing import Any, Iterable, Mapping

EXAMPLE_QUESTIONS = [
    "What's the weather in Chicago right now?",
    "5-day forecast for Seattle",
    "Will it snow tonight in Denver?",
    "How windy is it in Boston?",
    "Forecast for London in Celsius",
]

UNIT_LABELS = {
    "imperial": {"temperature": "°F", "wind_speed": "mph", "name": "US (°F, mph)"},
    "metric": {"temperature": "°C", "wind_speed": "m/s", "name": "Metric (°C, m/s)"},
}

UNAVAILABLE = "N/A"

PRECIP_NOTE = (
    "Precipitation chance is the highest probability of precipitation among the day's available "
    "3-hour forecast samples. Highs and lows are the warmest and coolest of those samples."
)
PARTIAL_NOTE = "Partial days are only covered by the forecast for the hours shown."

# OpenWeather icon codes (https://openweathermap.org/weather-conditions) -> emoji.
_ICONS = {
    "01d": "☀️", "01n": "🌙", "02d": "🌤️", "02n": "🌥️", "03": "☁️", "04": "☁️",
    "09": "🌧️", "10d": "🌦️", "10n": "🌧️", "11": "⛈️", "13": "❄️", "50": "🌫️",
}
_FALLBACK_ICON = "🌡️"

_TOPICS = {
    "precip": re.compile(
        r"\b(rain(y|ing|fall)?|snow(y|ing|fall)?|sleet|hail|drizzle|showers?|storms?|stormy|"
        r"thunder(storms?)?|lightning|umbrella|precip(itation)?|ice|icy|blizzard|wet|dry|"
        r"sunny|sun|clear|cloud(y|s)?|overcast|fog(gy)?)\b",
        re.IGNORECASE,
    ),
    "temperature": re.compile(
        r"\b(temp(erature)?s?|hot|cold|warm|cool|chilly|freez(e|ing)|frost|degrees?|feels?|"
        r"jacket|coat|sweater|hoodie|layers|shorts|gloves|scarf|high|low|heat)\b",
        re.IGNORECASE,
    ),
    "wind": re.compile(r"\b(wind(y|s)?|breez(e|y)|gusts?|gusty)\b", re.IGNORECASE),
    "humidity": re.compile(r"\b(humid(ity)?|muggy|sticky|dew)\b", re.IGNORECASE),
    "pressure": re.compile(r"\b(pressure|barometr(ic|e))\b", re.IGNORECASE),
}

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_TODAY = re.compile(r"\b(today|tonight|this\s+(morning|afternoon|evening)|right\s+now|now|later)\b", re.I)
_TOMORROW = re.compile(r"\btomorrow\b", re.I)
_WEEKEND = re.compile(r"\bweekend\b", re.I)
_WEEKDAY = re.compile(r"\b(" + "|".join(_WEEKDAYS) + r")s?\b", re.I)
_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "couple": 2, "few": 3, "several": 5}
_SPAN_DAYS = re.compile(
    r"\b(\d+|two|three|four|five|six)\s*-?\s*days?\b|"
    r"\b(?:next|coming)\s+(\d+|two|three|four|five|six|couple(?:\s+of)?|few|several)\s+days\b",
    re.IGNORECASE,
)
_SPAN_ALL = re.compile(r"\b(this\s+week|next\s+week|week|weekly|outlook|extended)\b", re.IGNORECASE)
_FULL_WINDOW_DAYS = 5  # OpenWeather's window: "5 day" (or more) means every available day


# --- formatting --------------------------------------------------------------------------------


def round_temp(value: Any) -> int | None:
    """Round half away from zero for display (avoids banker's rounding and ``-0``)."""
    if value is None:
        return None
    number = float(value)
    rounded = int(math.floor(abs(number) + 0.5)) * (1 if number >= 0 else -1)
    return rounded or 0


def format_temp(value: Any, units: str) -> str:
    rounded = round_temp(value)
    return UNAVAILABLE if rounded is None else f"{rounded}{UNIT_LABELS[units]['temperature']}"


def format_wind(value: Any, units: str) -> str:
    if value is None:
        return UNAVAILABLE
    label = UNIT_LABELS[units]["wind_speed"]
    # m/s values are small, so keep one decimal; mph reads fine as a whole number.
    return f"{float(value):.1f} {label}" if units == "metric" else f"{round_temp(value)} {label}"


def format_percent(value: Any) -> str:
    return UNAVAILABLE if value is None else f"{round_temp(value)}%"


def format_day(iso_date: str) -> str:
    """'2026-11-24' -> 'Tue, Nov 24' (no zero padding, platform-independent)."""
    d = date.fromisoformat(iso_date)
    return f"{d.strftime('%a')}, {d.strftime('%b')} {d.day}"


def format_clock(hh_mm: str, units: str) -> str:
    """'13:05' -> '1:05 PM' for US units, '13:05' for metric."""
    if units == "metric":
        return hh_mm
    t = datetime.strptime(hh_mm, "%H:%M")
    return f"{t.hour % 12 or 12}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def weather_icon(code: Any, *, daytime: bool = False) -> str:
    """Emoji for an OpenWeather icon code such as '04d'; a neutral icon when it's missing."""
    if not isinstance(code, str) or len(code) < 2:
        return _FALLBACK_ICON
    if daytime and code.endswith("n"):
        code = code[:-1] + "d"
    return _ICONS.get(code) or _ICONS.get(code[:2]) or _FALLBACK_ICON


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


# --- question analysis -------------------------------------------------------------------------


def question_topics(question: str) -> set[str]:
    return {name for name, pattern in _TOPICS.items() if pattern.search(question or "")}


def requested_span(question: str) -> int | None:
    """Number of days an explicit multi-day request asks for (0 = the whole window), else ``None``."""
    if match := _SPAN_DAYS.search(question or ""):
        raw = (match.group(1) or match.group(2) or "").lower().replace(" of", "").strip()
        days = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 0)
        if days >= 2:
            return 0 if days >= _FULL_WINDOW_DAYS else days
    if _SPAN_ALL.search(question or ""):
        return 0
    return None


def _requested_dates(question: str, available: list[date]) -> list[date] | None:
    """Dates a question is about, or ``None`` when it isn't about specific days.

    An empty list means it names days, but none of them are in ``available``.
    """
    if not available:
        return None
    first = available[0]
    wanted: set[date] = set()
    asked = False
    if _TODAY.search(question):
        asked = True
        wanted.add(first)
    if _TOMORROW.search(question):
        asked = True
        wanted.add(first + timedelta(days=1))
    weekdays = {_WEEKDAYS.index(m.lower()) for m in _WEEKDAY.findall(question)}
    if _WEEKEND.search(question):
        weekdays |= {5, 6}
    if weekdays:
        asked = True
        wanted |= {d for d in available if d.weekday() in weekdays}
    return sorted(wanted) if asked else None


# --- views -------------------------------------------------------------------------------------


def current_view(result: Mapping[str, Any], question: str) -> dict[str, Any]:
    units = result.get("units", "imperial")
    metrics = {
        "temperature": ("Temperature", format_temp(result.get("temperature"), units)),
        "feels_like": ("Feels like", format_temp(result.get("feels_like"), units)),
        "humidity": ("Humidity", format_percent(result.get("humidity_percent"))),
        "wind": ("Wind", format_wind(result.get("wind_speed"), units)),
    }
    # Pressure is shown only when the provider supplied it.
    if result.get("pressure_hpa") is not None:
        metrics["pressure"] = ("Pressure", f"{round_temp(result['pressure_hpa'])} hPa")

    topics = question_topics(question)
    keep = []
    if "temperature" in topics:
        keep += ["temperature", "feels_like"]
    if "humidity" in topics:
        keep.append("humidity")
    if "wind" in topics:
        keep.append("wind")
    if "pressure" in topics:
        keep.append("pressure")
    # Precipitation/sky questions are answered by the description alone.
    focused = bool(topics)
    if not focused:
        keep = list(metrics)

    observed = None
    if result.get("observed_local"):
        observed_at = datetime.fromisoformat(result["observed_local"])
        observed = f"Observed {format_clock(observed_at.strftime('%H:%M'), units)} local time"
        if result.get("utc_offset"):
            observed += f" (UTC{result['utc_offset']})"

    return {
        "kind": "current",
        "units": units,
        "location": result.get("location", ""),
        "description": _sentence_case(str(result.get("description", ""))),
        "icon": weather_icon(result.get("icon")),
        "metrics": [{"label": metrics[k][0], "value": metrics[k][1]} for k in keep if k in metrics],
        "observed": observed,
        "focused": focused,
    }


def _day_card(day: Mapping[str, Any], units: str) -> dict[str, Any]:
    first, last = day.get("first_sample_local"), day.get("last_sample_local")
    coverage = (
        f"{format_clock(first, units)}–{format_clock(last, units)} local" if first and last else UNAVAILABLE
    )
    return {
        "date": day["date"],
        "label": format_day(day["date"]),
        "partial": bool(day.get("partial")),
        "coverage": coverage,
        "icon": weather_icon(day.get("icon"), daytime=True),
        "conditions": _sentence_case(day.get("conditions") or "") or UNAVAILABLE,
        "also": list(day.get("other_conditions") or []),
        "high": format_temp(day.get("high"), units),
        "low": format_temp(day.get("low"), units),
        "precip": format_percent(day.get("precip_chance_percent")),
        "humidity": format_percent(day.get("avg_humidity_percent")),
        "wind": format_wind(day.get("max_wind_speed"), units),
    }


def _outlook(days: list[Mapping[str, Any]], units: str) -> str:
    """One or two data-only sentences summarizing the shown days."""
    highs = [round_temp(d["high"]) for d in days if d.get("high") is not None]
    lows = [round_temp(d["low"]) for d in days if d.get("low") is not None]
    t = UNIT_LABELS[units]["temperature"]
    parts = []
    conditions = Counter(d.get("conditions") for d in days if d.get("conditions"))
    if conditions:
        top, count = conditions.most_common(1)[0]
        if count * 2 > len(days):  # only claim "mostly" for a real majority of the shown days
            parts.append(f"Mostly {top}.")
    if highs and lows:
        span = lambda xs: f"{min(xs)}{t}" if min(xs) == max(xs) else f"{min(xs)}{t} to {max(xs)}{t}"  # noqa: E731
        parts.append(f"Highs {span(highs)}, lows {span(lows)}.")
    chances = [(d["precip_chance_percent"], d["date"]) for d in days if d.get("precip_chance_percent") is not None]
    if chances:
        best, best_date = max(chances, key=lambda c: c[0])
        if best >= 30:
            parts.append(f"Best chance of precipitation: {format_day(best_date)} ({best}%).")
        elif best == 0:
            parts.append("No precipitation in the forecast (0% chance each day).")
        else:
            parts.append(f"Precipitation chances stay at or below {best}%.")
    else:
        parts.append("Precipitation chances are unavailable.")
    return " ".join(parts)


def forecast_view(result: Mapping[str, Any], question: str) -> dict[str, Any]:
    units = result.get("units", "imperial")
    days = list(result.get("days") or [])
    available = [date.fromisoformat(d["date"]) for d in days]

    span = requested_span(question)
    wanted = None if span is not None else _requested_dates(question, available)
    if span:
        selected = days[:span]
    elif wanted is not None:
        selected = [d for d in days if date.fromisoformat(d["date"]) in wanted]
    else:
        selected = days
    day_not_in_window = wanted is not None and not selected
    if day_not_in_window:
        selected = days  # asked about a day outside the window: show what we have

    single = len(selected) == 1 and wanted is not None
    return {
        "kind": "forecast",
        "units": units,
        "location": result.get("location", ""),
        "layout": "single" if single else "multi",
        "outlook": "" if single else _outlook(selected, units),
        "days": [_day_card(d, units) for d in selected],
        "has_partial": any(d.get("partial") for d in selected),
        "focused": wanted is not None and not day_not_in_window,
        "day_not_in_window": day_not_in_window,
    }


# --- HTML for the day cards --------------------------------------------------------------------

DAY_CARDS_CSS = """
.wb-days { display: grid; grid-template-columns: repeat(auto-fit, minmax(7rem, 1fr)); gap: 0.6rem; margin: 0.25rem 0 0.75rem; }
.wb-days.wb-single { grid-template-columns: minmax(0, 20rem); }
.wb-day { background: rgba(250, 250, 250, 0.04); border: 1px solid rgba(250, 250, 250, 0.14); border-radius: 0.75rem;
          padding: 0.8rem 0.9rem; display: flex; flex-direction: column; gap: 0.2rem; min-width: 0; }
.wb-day.wb-partial { border-style: dashed; }
.wb-date { font-weight: 600; display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; }
.wb-badge { font-size: 0.7rem; font-weight: 600; padding: 0 0.45rem; border-radius: 999px; color: #ffbd45;
            background: rgba(255, 189, 69, 0.12); border: 1px solid rgba(255, 189, 69, 0.4); }
.wb-cover { font-size: 0.75rem; color: rgba(250, 250, 250, 0.6); }
.wb-icon { font-size: 2rem; line-height: 1.3; }
.wb-cond { font-size: 0.9rem; color: rgba(250, 250, 250, 0.85); overflow-wrap: anywhere; }
.wb-temps { font-size: 1.15rem; }
.wb-high { font-weight: 700; }
.wb-low { color: rgba(250, 250, 250, 0.65); }
.wb-precip, .wb-extra { font-size: 0.85rem; color: rgba(250, 250, 250, 0.8); }
.wb-na { color: rgba(250, 250, 250, 0.5); font-style: italic; }
@media (max-width: 640px) {
  .wb-day { padding: 0.65rem 0.7rem; }
  .wb-icon { font-size: 1.6rem; }
}
"""


def forecast_title(view: Mapping[str, Any]) -> str:
    """'Tue, Nov 24 – Sun, Nov 29 · Amarillo, US' (a date range reads clearer than an N-day count,
    since the provider's five-day window spans six calendar dates)."""
    days = view["days"]
    span = days[0]["label"] if len(days) == 1 else f"{days[0]['label']} – {days[-1]['label']}"
    return f"{span} · {view['location']}"


def _value_html(value: str) -> str:
    return f'<span class="wb-na" title="Not provided by the weather service">{UNAVAILABLE}</span>' if value == UNAVAILABLE else escape(value)


def day_cards_html(days: list[Mapping[str, Any]], *, single: bool = False) -> str:
    """Responsive grid of day cards (CSS grid wraps to fewer columns on narrow screens).

    All text is HTML-escaped; values come from the stored view, never from model text.
    """
    cards = []
    for day in days:
        partial = day.get("partial")
        badge = '<span class="wb-badge">Partial day</span>' if partial else ""
        cover = f'<div class="wb-cover">Covers {escape(day["coverage"])}</div>' if partial else ""
        extra = (
            f'<div class="wb-extra">Avg humidity {_value_html(day["humidity"])} · '
            f'Max wind {_value_html(day["wind"])}</div>'
            if single
            else ""
        )
        cards.append(
            f'<div class="wb-day{" wb-partial" if partial else ""}">'
            f'<div class="wb-date">{escape(day["label"])}{badge}</div>{cover}'
            f'<div class="wb-icon" role="img" aria-label="{escape(day["conditions"])}">{day["icon"]}</div>'
            f'<div class="wb-cond">{escape(day["conditions"])}</div>'
            f'<div class="wb-temps"><span class="wb-high">{_value_html(day["high"])}</span> '
            f'<span class="wb-low">/ {_value_html(day["low"])}</span></div>'
            f'<div class="wb-precip">💧 Precip chance {_value_html(day["precip"])}</div>{extra}'
            "</div>"
        )
    return f'<div class="wb-days{" wb-single" if single else ""}">{"".join(cards)}</div>'


def details_rows(days: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Rows for the expandable per-day details table."""
    return [
        {
            "Day": day["label"] + (" (partial)" if day.get("partial") else ""),
            "Conditions": day["conditions"] + (f" · also {', '.join(day['also'])}" if day.get("also") else ""),
            "Precip chance": day["precip"],
            "Avg humidity": day["humidity"],
            "Max wind": day["wind"],
            "Forecast hours": day["coverage"],
        }
        for day in days
    ]


def build_views(question: str, tool_calls: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Display views for the successful tool results of one turn (failed lookups are skipped)."""
    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in tool_calls:
        result = call.get("result") or {}
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            continue
        if call.get("name") == "get_current_weather":
            view = current_view(result, question)
        elif call.get("name") == "get_weather_forecast":
            view = forecast_view(result, question)
        else:
            continue
        key = repr(sorted(view.items()))
        if key not in seen:  # the model sometimes repeats an identical lookup
            seen.add(key)
            views.append(view)
    return views
