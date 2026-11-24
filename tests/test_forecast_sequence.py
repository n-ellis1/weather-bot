"""Regression: "whats the weather in amarillo" -> "what about tomorrow" ->
"give me the 5 day forecast for amarillo texas".

Live, Grok answered the third turn from the earlier reply text without calling a tool, so the
reply had no fresh data (and invented highs) and no forecast cards were shown. Forcing a tool call
with ``tool_choice`` hung on xAI with that history, so the agent now fetches fresh data itself when
a data request is about a city already looked up in the conversation (``prefetch_plan``).

``LazyModel`` reproduces the live behaviour: it looks a city up only on the first turn, and after
that answers from the conversation text unless this turn already has tool results.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest

import weather_tools
from chat_agent import lookups_from, plan_lookup, run_turn
from presentation import build_views

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TURNS = ["whats the weather in amarillo", "what about tomorrow", "give me the 5 day forecast for amarillo texas"]
MEMORY_ANSWER = "The next five days stay mild, with highs in the mid-50s to low 60s."

CURRENT_PAYLOAD = {
    "name": "Amarillo", "sys": {"country": "US"}, "dt": 1795546800, "timezone": -21600,
    "main": {"temp": 44.2, "feels_like": 40.1, "humidity": 62, "pressure": 1019},
    "wind": {"speed": 8.1}, "weather": [{"description": "overcast clouds", "icon": "04d"}],
}


def _forecast_payload():
    start = int(datetime(2026, 11, 24, 21, tzinfo=timezone.utc).timestamp())  # 15:00 local (UTC-6)
    samples = []
    for i in range(40):
        samples.append({
            "dt": start + i * 3 * 3600,
            "main": {"temp": 40.0 + (i % 8) * 2, "humidity": 50 + i % 5},
            "weather": [{"description": "clear sky" if i % 3 else "few clouds", "icon": "01d" if i % 3 else "02d"}],
            "wind": {"speed": 5.0 + i % 4},
            "pop": (i % 10) / 10,
        })
    return {"city": {"name": "Amarillo", "country": "US", "timezone": -21600}, "list": samples}


FORECAST_PAYLOAD = _forecast_payload()


def _http(url, params, timeout):
    resp = MagicMock(status_code=200)
    resp.json.return_value = FORECAST_PAYLOAD if url == weather_tools.FORECAST_URL else CURRENT_PAYLOAD
    return resp


class LazyModel:
    """Looks a city up only when the conversation has no earlier replies; otherwise answers from memory."""

    def __init__(self):
        self.memory_answers = 0

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        model = self

        class _Bound:
            def invoke(self, messages):
                if isinstance(messages[-1], ToolMessage) or tool_choice == "none":
                    return AIMessage(content="Here's what the tools returned for Amarillo, US.")
                if any(isinstance(m, AIMessage) for m in messages):  # earlier replies exist: the bug
                    model.memory_answers += 1
                    return AIMessage(content=MEMORY_ANSWER)
                return AIMessage(content="", tool_calls=[
                    {"name": "get_current_weather", "args": {"city": "Amarillo"}, "id": "c1", "type": "tool_call"}])

        return _Bound()


@pytest.fixture(autouse=True)
def owm_key(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-owm-key")


def test_sequence_agent_level_every_turn_gets_fresh_data_and_the_right_view():
    model, history, all_views, replies = LazyModel(), [], [], []
    with patch.object(weather_tools.requests, "get", side_effect=_http) as get:
        for prompt in TURNS:
            result = run_turn(model, history, prompt)
            all_views.append(build_views(prompt, result["tool_calls"]))
            replies.append(result["reply"])
            # Stored the way app.answer stores assistant messages.
            history += [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": result["reply"], "lookups": lookups_from(result["tool_calls"])}]

    assert model.memory_answers == 0
    assert MEMORY_ANSWER not in replies
    assert [c.args[0] for c in get.call_args_list] == [
        weather_tools.CURRENT_WEATHER_URL, weather_tools.FORECAST_URL, weather_tools.FORECAST_URL,
    ]
    assert [c.kwargs["params"]["q"] for c in get.call_args_list] == ["Amarillo", "Amarillo", "Amarillo,TX,US"]
    current, tomorrow, five_day = (v[0] for v in all_views)
    assert current["kind"] == "current"
    assert tomorrow["kind"] == "forecast" and tomorrow["layout"] == "single" and len(tomorrow["days"]) == 1
    # The explicit 5-day request shows the whole window even though the previous turn was one day.
    assert five_day["kind"] == "forecast" and five_day["layout"] == "multi"
    assert [d["date"] for d in five_day["days"]] == [
        "2026-11-24", "2026-11-25", "2026-11-26", "2026-11-27", "2026-11-28", "2026-11-29",
    ]
    assert five_day["days"][0]["partial"] and five_day["days"][-1]["partial"]


def test_without_stored_lookups_a_named_city_is_still_looked_up():
    # History without lookups (e.g. from before they were stored): the city named in the message is
    # enough for the agent to fetch it itself, so the lazy model never answers from memory.
    model = LazyModel()
    history = [{"role": "user", "content": TURNS[0]}, {"role": "assistant", "content": "It's 44°F in Amarillo, US."}]
    with patch.object(weather_tools.requests, "get", side_effect=_http) as get:
        result = run_turn(model, history, TURNS[2])
    assert model.memory_answers == 0 and get.call_count == 1
    assert get.call_args.args[0] == weather_tools.FORECAST_URL and get.call_args.kwargs["params"]["q"] == "Amarillo,TX,US"
    assert result["verified"] is True and result["lookup_failed"] is False
    assert MEMORY_ANSWER not in result["reply"]


def test_place_less_follow_up_with_no_stored_city_never_shows_a_memory_answer():
    # No stored lookups and no place in the message: nothing to fetch. The lazy model answers from
    # memory, and the final check blocks it.
    model = LazyModel()
    history = [{"role": "user", "content": TURNS[0]}, {"role": "assistant", "content": "It's 44°F in Amarillo, US."}]
    with patch.object(weather_tools.requests, "get", side_effect=_http) as get:
        result = run_turn(model, history, "what about tomorrow")
    assert model.memory_answers == 1 and get.call_count == 0
    assert result["reply"] != MEMORY_ANSWER and result["verified"] is False


def test_sequence_in_the_app_shows_card_then_day_card_then_full_forecast(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    st.cache_resource.clear()  # the app caches its model; make sure it builds this fake one
    model = LazyModel()
    with patch("chat_agent.build_model", return_value=model), patch.object(weather_tools.requests, "get", side_effect=_http):
        at = AppTest.from_file(APP, default_timeout=30).run()
        for prompt in TURNS:
            at.chat_input[0].set_value(prompt).run()

    assert not at.exception
    assert model.memory_answers == 0
    replies = at.session_state.messages[1::2]
    assert [r["lookups"] for r in replies] == [
        [{"tool": "get_current_weather", "city": "Amarillo", "location": "Amarillo, US"}],
        [{"tool": "get_weather_forecast", "city": "Amarillo", "location": "Amarillo, US"}],
        [{"tool": "get_weather_forecast", "city": "Amarillo,TX,US", "location": "Amarillo, US"}],
    ]
    assert [[v["kind"] for v in r["views"]] for r in replies] == [["current"], ["forecast"], ["forecast"]]
    assert [r["views"][0].get("layout") for r in replies] == [None, "single", "multi"]
    assert len(replies[2]["views"][0]["days"]) == 6

    card_html = [m.value for m in at.markdown if 'class="wb-days' in m.value]
    assert len(card_html) == 2
    assert card_html[0].count('class="wb-date"') == 1 and "wb-single" in card_html[0]
    assert card_html[1].count('class="wb-date"') == 6 and card_html[1].count("Partial day") == 2
    assert len(at.expander) == 1 and len(at.table) == 1  # details table for the 5-day view only
    assert list(at.table[0].value.columns) == ["Conditions", "Precip chance", "Avg humidity", "Max wind", "Forecast hours"]
    assert any("highest probability of precipitation" in c.value for c in at.caption)


# --- a new city in an ongoing chat ------------------------------------------------------------------

DENVER = "What's the 5 day forecast for Denver?"


def _http_by_city(url, params, timeout):
    """Like ``_http``, but the payload is for whichever city was queried (Amarillo or Denver)."""
    city = params["q"].split(",")[0].strip().title()
    resp = MagicMock(status_code=200)
    if url == weather_tools.FORECAST_URL:
        resp.json.return_value = FORECAST_PAYLOAD | {"city": FORECAST_PAYLOAD["city"] | {"name": city, "timezone": -25200}}
    else:
        resp.json.return_value = CURRENT_PAYLOAD | {"name": city}
    return resp


def test_new_city_after_the_amarillo_turns_fetches_and_renders_denver(monkeypatch):
    """Regression: after the three Amarillo turns, a model that skips the tool call still gets Denver
    data, because the agent looks the newly named city up itself, and the app renders its forecast grid."""
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    st.cache_resource.clear()
    model = LazyModel()
    with patch("chat_agent.build_model", return_value=model), \
            patch.object(weather_tools.requests, "get", side_effect=_http_by_city) as get:
        at = AppTest.from_file(APP, default_timeout=30).run()
        for prompt in [*TURNS, DENVER]:
            at.chat_input[0].set_value(prompt).run()

    assert not at.exception
    assert model.memory_answers == 0  # never answered from memory, Denver included
    # Denver was fetched (forecast endpoint, imperial by default), after the three Amarillo lookups.
    queries = [(c.args[0], c.kwargs["params"]["q"], c.kwargs["params"]["units"]) for c in get.call_args_list]
    assert queries[-1] == (weather_tools.FORECAST_URL, "Denver", "imperial")
    assert [q for _, q, _ in queries[:3]] == ["Amarillo", "Amarillo", "Amarillo,TX,US"]

    denver = at.session_state.messages[-1]
    assert denver.get("lookup_failed") is not True and len(at.warning) == 0
    assert [v["kind"] for v in denver["views"]] == ["forecast"]
    view = denver["views"][0]
    assert view["location"] == "Denver, US" and view["layout"] == "multi" and len(view["days"]) == 6
    assert denver["lookups"] == [{"tool": "get_weather_forecast", "city": "Denver", "location": "Denver, US"}]

    # The rendered page: the Denver forecast grid is the last card grid, with all six day cards.
    grids = [m.value for m in at.markdown if 'class="wb-days' in m.value]
    assert len(grids) == 3  # tomorrow (Amarillo), 5-day Amarillo, 5-day Denver
    assert grids[-1].count('class="wb-date"') == 6 and grids[-1].count("Partial day") == 2
    titles = [m.value for m in at.markdown if m.value.startswith("#### 🗓️")]
    assert titles[-1].endswith("· Denver, US")

    # A place-less follow-up now refers to Denver, the newest city.
    with patch("chat_agent.build_model", return_value=model), \
            patch.object(weather_tools.requests, "get", side_effect=_http_by_city) as get:
        at.chat_input[0].set_value("what about tomorrow").run()
    assert get.call_args.kwargs["params"]["q"] == "Denver"
    assert at.session_state.messages[-1]["views"][0]["location"] == "Denver, US"


def test_ambiguous_city_asks_then_uses_the_answer(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    st.cache_resource.clear()
    model = LazyModel()
    with patch("chat_agent.build_model", return_value=model), \
            patch.object(weather_tools.requests, "get", side_effect=_http_by_city) as get:
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.chat_input[0].set_value("whats the weather in amarillo").run()
        at.chat_input[0].set_value("What's the forecast for Portland?").run()
        asked = at.session_state.messages[-1]
        calls_after_question = get.call_count
        at.chat_input[0].set_value("Oregon").run()

    assert asked["content"] == ("There's more than one Portland. Did you mean Portland, Oregon or Portland, Maine? "
                                "Reply with the state or country.")
    assert "views" not in asked and calls_after_question == 1  # asking made no lookup
    assert get.call_args.args[0] == weather_tools.FORECAST_URL
    assert get.call_args.kwargs["params"]["q"] == "Portland,OR,US"
    assert at.session_state.messages[-1]["views"][0]["location"] == "Portland, US"


# --- plan_lookup ----------------------------------------------------------------------------------

AMARILLO = [{"role": "user", "content": "whats the weather in amarillo"},
            {"role": "assistant", "content": "44°F.", "lookups": [
                {"tool": "get_current_weather", "city": "Amarillo", "location": "Amarillo, US"}]}]
TWO_CITIES = AMARILLO + [{"role": "user", "content": "and in Paris?"},
                         {"role": "assistant", "content": "12°C.", "lookups": [
                             {"tool": "get_current_weather", "city": "Paris,FR", "location": "Paris, FR"}]}]


def _plan(prompt, history, reason):
    plan = plan_lookup(prompt, history, reason)
    if plan is None:
        return None
    return (plan["tool"], plan["city"]) if plan["action"] == "fetch" else ("clarify", plan["reply"])


@pytest.mark.parametrize(
    ("prompt", "history", "reason", "expected"),
    [
        # A state-qualified name is looked up precisely, not by reusing the bare earlier query.
        ("give me the 5 day forecast for amarillo texas", AMARILLO, "weather", ("get_weather_forecast", "Amarillo,TX,US")),
        ("what about tomorrow", AMARILLO, "follow_up", ("get_weather_forecast", "Amarillo")),
        ("and right now?", AMARILLO, "follow_up", ("get_current_weather", "Amarillo")),
        ("Is it still cold in Amarillo?", AMARILLO, "weather", ("get_current_weather", "Amarillo")),
        ("Is Amarillo still cold?", AMARILLO, "weather", ("get_current_weather", "Amarillo")),  # named without "in"
        ("forecast for this weekend", AMARILLO, "weather", ("get_weather_forecast", "Amarillo")),
        ("what about tomorrow", TWO_CITIES, "follow_up", ("get_weather_forecast", "Paris,FR")),  # most recent city
        ("5 day forecast for amarillo", TWO_CITIES, "weather", ("get_weather_forecast", "Amarillo")),  # named city
        # New cities are looked up directly and take precedence over the previous city.
        ("What's the 5 day forecast for Denver?", AMARILLO, "weather", ("get_weather_forecast", "Denver")),
        ("whats the weather in amarillo", [], "weather", ("get_current_weather", "Amarillo")),
        ("weather in Paris, France", AMARILLO, "weather", ("get_current_weather", "Paris,FR")),
        ("weather in Portland, Maine", AMARILLO, "weather", ("get_current_weather", "Portland,ME,US")),
        # Ambiguous bare names get a question instead of a lookup.
        ("What's the forecast for Portland?", AMARILLO, "weather",
         ("clarify", "There's more than one Portland. Did you mean Portland, Oregon or Portland, Maine? "
                     "Reply with the state or country.")),
        ("What causes thunder?", AMARILLO, "weather", None),  # general question, no place, not a follow-up
        ("What's the forecast?", [], "weather", None),  # no place anywhere: the check asks for the city
        ("Weather in kelvin?", [], "weather", None),  # a unit, not a place
        ("thanks!", AMARILLO, "small_talk", None),
        ("Tell me a joke", AMARILLO, "off_topic", None),
    ],
)
def test_plan_lookup(prompt, history, reason, expected):
    assert _plan(prompt, history, reason) == expected


def test_answer_to_an_ambiguity_question_picks_the_option():
    question = plan_lookup("What's the forecast for Portland?", AMARILLO, "weather")
    history = AMARILLO + [{"role": "user", "content": "What's the forecast for Portland?"},
                          {"role": "assistant", "content": question["reply"],
                           "ambiguous_place": question["ambiguous_place"]}]
    assert _plan("Oregon", history, "follow_up") == ("get_weather_forecast", "Portland,OR,US")
    assert _plan("the one in Maine", history, "follow_up") == ("get_weather_forecast", "Portland,ME,US")
    # An answer that picks nothing falls back to normal planning (here: a follow-up for Amarillo).
    assert _plan("what about tomorrow", history, "follow_up") == ("get_weather_forecast", "Amarillo")

def test_lookups_from_keeps_only_successful_weather_calls():
    calls = [
        {"name": "get_current_weather", "args": {"city": "Amarillo", "units": "imperial"},
         "result": {"ok": True, "location": "Amarillo, US"}},
        {"name": "get_weather_forecast", "args": {"city": "Nowhere"}, "result": {"ok": False, "error": "City not found"}},
        {"name": "run_shell", "args": {"city": "x"}, "result": {"ok": True}},
    ]
    assert lookups_from(calls) == [{"tool": "get_current_weather", "city": "Amarillo", "location": "Amarillo, US"}]
