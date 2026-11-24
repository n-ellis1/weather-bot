# Weather-Bot

A Streamlit chat assistant that answers weather questions only. It uses Grok 4.7 (through LangChain's `langchain-xai` integration) and two OpenWeather tools: current conditions and the five-day forecast. Replies come as a short text answer, then a card or day-card grid built directly from the weather data.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or `pip`
- An xAI API key (for Grok) and an OpenWeather API key (the free tier covers both endpoints used here)

## Install dependencies

With uv (installs the app and dev dependencies from `uv.lock` into `.venv`):

```sh
uv sync
```

With pip instead:

```sh
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest   # only needed to run the tests
```

`requirements.txt` is generated from `uv.lock` (`uv export --no-hashes --no-dev -o requirements.txt`). Regenerate it after changing dependencies.

## Configure your API keys

Keys are read from a local `.env` file next to `app.py`. That file is git-ignored; `.env.example` is the committed template.

1. Copy the template:

   ```sh
   # macOS/Linux/Git Bash
   cp .env.example .env
   ```

   ```powershell
   # Windows PowerShell
   Copy-Item .env.example .env
   ```

2. Open `.env` and fill in your own keys. The values below are placeholders, not real keys:

   ```dotenv
   OPENWEATHER_API_KEY=your-openweather-api-key
   XAI_API_KEY=your-xai-api-key
   ```

Never commit `.env` or paste real keys into issues, logs, or chat. Variables already set in your shell take precedence over `.env`.

## Run the app

```sh
uv run streamlit run app.py
```

Then open http://localhost:8501. (With pip, activate the virtual environment and run `streamlit run app.py`.) The app loads `.env` relative to `app.py`, so it works from any working directory. It starts in dark mode (`.streamlit/config.toml`).

In the sidebar you can switch units (US °F/mph is the default, or metric °C/m/s), click example questions, and clear the chat. Asking for a unit in a message ("…in Celsius?") overrides the setting for that reply only.

If you change code while the server is running, restart it: Streamlit re-runs `app.py` but keeps imported modules such as `presentation.py` loaded.

## Run the tests

```sh
uv run pytest -q
```

(With pip: `python -m pytest -q`.) The suite currently has 393 tests. It doesn't use the network or real keys. Grok is replaced by scripted fake models, OpenWeather HTTP calls are mocked, and the app is driven through Streamlit's `AppTest`. Keys are fake `test-…` values set per test. The only expected warnings are third-party `langsmith` deprecation warnings.

| File | Covers |
| --- | --- |
| `tests/test_current_weather.py`, `tests/test_forecast.py` | The two OpenWeather tools: parsing, daily aggregation, and error mapping |
| `tests/test_safeguards.py` | Local screening: refusals, clarifications, and follow-ups |
| `tests/test_chat_agent.py` | The tool loop, allowlist, call caps, and units |
| `tests/test_forecast_sequence.py`, `tests/test_places.py` | Lookups the agent makes itself, and place parsing |
| `tests/test_grounding.py` | The final check before live weather is shown |
| `tests/test_presentation.py`, `tests/test_app.py` | Cards, tables, and the Streamlit page |

## How it works

```
Streamlit (app.py)
  │  user message + chat history (st.session_state)
  ▼
safeguards.screen_request ──► local refusal or clarifying question (no Grok, no OpenWeather)
  │ allowed
  ▼
chat_agent.run_turn
  ├─ plan_lookup: the agent may call a weather tool itself first
  ├─ Grok 4.7 via LangChain ChatXAI, with exactly two tools bound
  │     ├─ get_current_weather  ──► OpenWeather /data/2.5/weather
  │     └─ get_weather_forecast ──► OpenWeather /data/2.5/forecast
  └─ verify_reply: final check of the reply against this turn's tool results
  ▼
presentation.build_views ──► current-conditions card or forecast day cards
```

- **Streamlit** (`app.py`) is the UI. It keeps the conversation in `st.session_state.messages`, screens each message, calls the agent, and renders the reply followed by cards built from the structured tool results. Card numbers come from the tool data, not from the model's text.
- **LangChain** provides the model wrapper (`ChatXAI` from `langchain-xai`), the `@tool` definitions, and the message types. `chat_agent.run_turn` is a small, explicit tool-calling loop. It doesn't use a generic agent executor, so the allowlist and limits are enforced in our code.
- **Grok 4.7** (`grok-4.7`, temperature 0, 60 s timeout, 2 retries) decides which tool to call and writes the short reply. Its system prompt restricts it to weather, tells it to base every weather fact on this turn's tool results, and tells it to quote tool errors instead of guessing.
- **The two OpenWeather tools** (`weather_tools.py`):
  - `get_current_weather(city, units)` returns the resolved location, temperature, feels-like, humidity, wind, pressure, conditions, and local observation time.
  - `get_weather_forecast(city, units)` turns the 5-day / 3-hour forecast into one entry per local day: high and low, dominant conditions, highest precipitation chance, average humidity, maximum wind, and whether the day is partial.

  Both always return a dict: `ok: true` with data, or `ok: false` with a user-safe `error` message.

Grok doesn't always call a tool when earlier weather replies are in the chat, and forcing it with `tool_choice` hangs on xAI. So for a clear current-weather or forecast request, `plan_lookup` runs the right tool itself before the model answers. It does this when the city is named in the message, is a city mentioned earlier, or is the most recent city for a follow-up. Common forms such as "amarillo texas" or "Paris, France" are rewritten into OpenWeather's query format (`places.py`), and a short list of ambiguous names such as "Portland" gets a "which one?" question instead.

## Safeguards

These are the protections the code actually implements. They are layered heuristics, not guarantees. See [Known limitations](#known-limitations).

### Local weather-only screening (`safeguards.py`)

Every message is classified in plain Python **before** the app builds or calls Grok, and before any OpenWeather request. Text is NFKC-normalized, zero-width characters are removed, and whitespace is collapsed before matching. The checks run in this order:

1. Requests for secrets or the bot's instructions, and override attempts → **declined** (see the next section).
2. Local actions are **declined** even if they mention weather. These include files and folders, shell commands, opening or installing apps, email or text messages, and reminders or calendar items.
3. Coding requests → **declined**.
4. Messages with weather wording, including clothing and outdoor-plan words such as "umbrella" or "picnic" → **allowed**.
5. Clearly off-topic tasks with no weather wording, such as jokes, recipes, math, trivia, news, or web search → **declined**.
6. A short answer to one of the bot's clarifying questions → **allowed**.
7. Short follow-ups ("What about tomorrow?", "And in London?") → **allowed**, but only if a recent message (last 8) was a weather question from the user. The bot's own replies never count as weather context, so a refusal that mentions weather can't unlock follow-ups.
8. Greetings, thanks, and "what can you do?" → **allowed**.
9. Anything else gets a **local clarifying question** asking for current conditions or the forecast, and a city.

Declined and clarified messages never reach Grok or OpenWeather.

### Prompt-injection refusals

These are declined locally with a fixed reply, even if the message also asks about weather:

- Requests for API keys, tokens, passwords, credentials, `.env`, or environment variables
- Requests for the system, internal, or hidden prompt, or for "your instructions/rules"
- Override attempts such as "ignore previous instructions", "disregard your rules", "print everything above", "you are now…", "pretend you're…", "developer mode", and "jailbreak"

Anything that gets past the screen still meets Grok's system prompt. It says the scope can't be changed by user messages, and that the bot must refuse to reveal keys, credentials, environment variables, or the prompt.

### Tool allowlist and call limits (`chat_agent.py`)

- Only `get_current_weather` and `get_weather_forecast` are bound to the model. If the model asks for any other tool name, it gets an "Unknown tool" error and nothing runs.
- Each user message allows at most **4 tool calls** (`MAX_TOOL_CALLS_PER_TURN`) across at most **3 model rounds** (`MAX_TOOL_ROUNDS`). Calls over the cap aren't run. Once the rounds are used up, the model is called one last time with tools disabled (`tool_choice="none"`), so a turn always ends.
- Repeated lookups of the same tool and place in one turn reuse the earlier result instead of sending another request.
- Bad arguments or tool exceptions return a generic "invalid arguments" error. Raw exception text is never shown.
- Units are set by the app (the sidebar setting or an explicit request in the message) and override whatever units the model passes.

### Credential handling

- Keys exist only in the process environment, loaded from the git-ignored `.env`. Only `.env.example` with empty values is committed. `.streamlit/secrets.toml` is also git-ignored.
- `ChatXAI` reads `XAI_API_KEY` itself. Our code only checks that it's set. The OpenWeather key is read inside the HTTP helper and sent only to OpenWeather, as the `appid` query parameter its API requires.
- Neither key is in the system prompt or in tool results, so the model never sees them.
- Network errors are mapped to fixed messages such as "Could not reach the weather service". Exception text, which can include the request URL and therefore the key, is never shown to the model or user. Model or API failures in the app show only the exception type.
- If `XAI_API_KEY` is missing, the app shows a "not configured" message and doesn't call the model.

### Verification before live weather is shown (`chat_agent.verify_reply`)

For requests about actual current or forecast weather (not general questions like "What causes thunder?"), the reply is checked before display:

- The current turn must have a **successful lookup** for the requested place. A forecast request must have a forecast lookup. Tool results from earlier turns don't count.
- The place is matched against the **location OpenWeather returned**, not the query that was sent. A lookup that resolved to a different city is never shown as a card or remembered as the current city.
- If there's no matching lookup, the model's text is **discarded**. The user sees a ⚠️ warning that quotes the tool's error, or says no verified data was available, instead of numbers the model may have guessed. If no place is known at all, the bot asks "Which city should I check the weather for?"
- Even with a matching lookup, any measurement in the reply (a number with °, °F, °C, %, mph, m/s, km/h, or hPa) must be within 1 of a value the tools returned this turn. Otherwise the reply becomes a data-only line ("Here's the latest forecast for Denver, US."), and the cards show the data.

## Known limitations

- **Screening is regex-based, not a classifier.** Weather words let things through: "Is the new iPhone cool?" reaches Grok, and only the system prompt stops it. A short message right after a weather answer, or after a clarifying question, is also sent to Grok if it doesn't hit a decline pattern. Reworded injections that include weather words rely on the system prompt alone.
- **Over-blocking and over-clarifying.** Mixed requests such as "Email me the forecast" or "Set a reminder if it rains" are declined, and any mention of "files", "folder", or "desktop" is declined. Weather questions with no weather word and no recent context ("Should I bike to work tomorrow in Austin?") get a clarifying question first. Screening is English-only.
- **The verification check is heuristic too.** It only checks numbers that have units. Vague phrases like "mid-50s" aren't checked, but they're only shown when the turn has a matching successful lookup. Numbers in answers to general weather questions aren't checked. A general question that names a place ("How does humidity work in Florida?") is treated as a live request and fails if the model doesn't look it up.
- **Place handling.**
  - The agent only spots a city named after "in/for/at/near/around", or one already discussed. Other new cities ("Denver forecast?") are left to the model, and the check catches a skipped lookup.
  - Ambiguity uses a curated list of 22 names, not live geocoding. Other bare names get OpenWeather's top match ("Paris" → Paris, FR).
  - Two-letter codes that are also US states are read as states, so "Toronto, CA" becomes `Toronto,CA,US` and fails.
  - Aliases such as "NYC" → New York are flagged as the wrong city. Typos get OpenWeather's "city not found".
  - Cards show city and country only ("Portland, US"), not the state.
- **Forecast data.** The forecast covers OpenWeather's 5-day / 3-hour window, which usually spans six calendar dates with partial first and last days. Daily highs and lows are the max and min of the 3-hour samples, not true daily extremes. A "tonight" question shows the whole day's card.
- **Units.** Explicit unit requests don't cover Kelvin or km/h; those use the sidebar setting. Earlier replies stay in the units they were given in.
- **Model behavior.** Grok sometimes skips tool calls when there's chat history, and xAI hangs when a tool call is forced with `tool_choice`. The agent's own lookups and the final check cover the common cases, so the user gets a warning rather than invented numbers. They don't make Grok more reliable.
- **Session and deployment.** Chat history lives only in the browser session and is sent to Grok in full on each turn. The app has no user authentication or rate limiting of its own. Anyone who can reach the server uses your API keys and quota. It's meant to run locally.
- **Testing.** The automated tests use fakes and mocks only. Live checks with real keys and a browser have been run by hand for the main flows. Some scope cases haven't been tried against live Grok: a live follow-up, a clarification answer, and an off-topic prompt that contains a weather word.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit page: sidebar, chat history, rendering, and error handling |
| `chat_agent.py` | Grok model, system prompt, tool loop, the agent's own lookups, and the final check |
| `weather_tools.py` | The two OpenWeather LangChain tools |
| `safeguards.py` | Local request screening and fixed refusal/clarification text |
| `places.py` | Place-name normalization and the ambiguous-city list (no network calls) |
| `presentation.py` | Turns tool results into card and table views |
| `tests/` | pytest suite (mocked HTTP and fake models) |
