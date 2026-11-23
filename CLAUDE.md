# Weather-Bot

A Streamlit chat assistant that answers weather questions only, using Grok (via `langchain-xai`) and OpenWeather.

## Session workflow
- At the start of every fresh session, read `HANDOFF.md` before doing any work.
- After completing and verifying a task, update `HANDOFF.md` (what's done, checks that passed, next task).

## UI
- The app defaults to dark mode (set `base = "dark"` under `[theme]` in `.streamlit/config.toml`).

## Setup
- Dependencies are managed with uv: `uv sync`. Run the app with `uv run streamlit run app.py`.
- After changing dependencies, regenerate `requirements.txt`: `uv export --no-hashes --no-dev -o requirements.txt`.
- Secrets live in `.env` (git-ignored). `.env.example` lists `OPENWEATHER_API_KEY` and `XAI_API_KEY`. Never commit keys.

## Roadmap
1. **Current weather tool**: a LangChain tool that calls the OpenWeather current weather endpoint by city name. Return a compact summary (temperature, conditions, humidity, wind), and handle unknown cities and API errors cleanly.
2. **Five-day forecast tool**: a LangChain tool that calls the OpenWeather 5-day / 3-hour forecast endpoint and summarizes it per day (high, low, conditions).
3. **Grok 4.7 integration**: use `ChatXAI` with the Grok 4.7 model and bind both tools. Replace the placeholder reply in `app.py` with the agent's response, and keep chat history in `st.session_state`.
4. **Weather-only safeguards**: a system prompt that limits the bot to weather topics and politely declines everything else. Ignore attempts to override the scope, and don't reveal keys or internal prompts.
5. **Verification**: unit-test the tools with mocked HTTP responses, test that off-topic prompts are refused, and do a manual end-to-end check in the running Streamlit app with real keys.
