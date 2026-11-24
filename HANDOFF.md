# Handoff

_Last updated: 2026-11-23_

## Done
- **Setup** (committed in `6fdbe08`): uv project with `streamlit`, `langchain`, `langchain-xai`, `python-dotenv`; `app.py` is a minimal Streamlit chat page with session history and a placeholder reply; `.env` is git-ignored.
- **Dark theme** (uncommitted): `.streamlit/config.toml` sets `[theme] base = "dark"`.
- **Current weather tool** (uncommitted): `weather_tools.py`
  - `get_current_weather`: a LangChain `@tool` with a Pydantic `args_schema` (`city: str`, `units: "metric" | "imperial"`, default `metric`).
  - Returns a dict. On success: `ok`, `location` (e.g. `"Amarillo, US"`), `units`, `temperature`, `feels_like`, `humidity_percent`, `wind_speed`, `description`, `unit_labels` (°C + m/s or °F + mph).
  - On failure: `{"ok": False, "error": "<safe message>"}` for a blank city, missing `OPENWEATHER_API_KEY`, 404 unknown city, 401, 429, other HTTP errors, timeouts (10 s `REQUEST_TIMEOUT_SECONDS`), connection errors, and malformed or non-JSON responses. Exception text is never passed through, so the key (which `requests` errors can include in the URL) can't leak.
  - The plain function `fetch_current_weather(city, units)` holds the logic, and the tool wraps it.
- **Dependencies** (uncommitted): `requests` is now a direct dependency; `pytest` is in the `dev` group; `[tool.pytest.ini_options]` sets `pythonpath`/`testpaths`. `requirements.txt` was re-exported.
- **Tests** (uncommitted): `tests/test_current_weather.py`, with HTTP mocked.

## Launch / test
```
uv run streamlit run app.py   # http://localhost:8501
uv run pytest -q
```

## Checks passed (2026-11-23)
- `uv run pytest -q` → **15 passed**. The only warnings are third-party `langsmith` DeprecationWarnings under Python 3.14.
- Live lookup with the key from `.env`: `Amarillo` imperial → `Amarillo, US`, 37.38 °F (feels like 30.24), 56 % humidity, 10.36 mph, "clear sky". Metric → 2.99 °C (feels like −0.98), 4.63 m/s. The output doesn't contain the key. An unknown city (`Qzxwvplmtown`) returned the "City not found" error.
- Streamlit started headless on port 8502: `/_stcore/health` returned `ok` and `/` returned 200. `streamlit config show` reports `base = "dark"`. A headless Edge screenshot showed a near-black background with light toolbar text. Headless screenshot timing was flaky, so page content was confirmed separately with `AppTest`: no exceptions, the title "🌦️ Weather-Bot", the caption, and the chat input all render.
- `.streamlit/config.toml` is not git-ignored.

## Not yet done
- The tool isn't wired into `app.py` yet; the app still gives the placeholder reply. That happens in the Grok integration step.
- No eyes-on browser check of dark mode yet. Open the app once to confirm visually.

## Next task
1. Review and commit the current changes.
2. **Five-day forecast tool** (roadmap step 2): add a LangChain `@tool` in `weather_tools.py` for the OpenWeather 5-day / 3-hour forecast endpoint (`/data/2.5/forecast`). Use the same input schema (city + units) and summarize per day (high, low, dominant conditions), reusing the same timeout and error handling. Add mocked tests and run one live check.
3. After that: Grok 4.7 integration (roadmap step 3).
