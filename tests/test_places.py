"""Place normalization, ambiguity, and in-turn lookup de-duplication."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

import weather_tools
from chat_agent import run_turn
from places import ambiguity_question, normalize_place, pick_option, same_place_query


@pytest.mark.parametrize(
    ("text", "query", "qualified"),
    [
        ("Denver", "Denver", False),
        ("denver", "Denver", False),
        ("amarillo texas", "Amarillo,TX,US", True),
        ("Amarillo, TX", "Amarillo,TX,US", True),
        ("amarillo tx", "Amarillo,TX,US", True),
        ("Paris, France", "Paris,FR", True),
        ("paris france", "Paris,FR", True),
        ("Paris, Texas", "Paris,TX,US", True),
        ("London, England", "London,GB", True),
        ("new york", "New York", False),
        ("New York, New York", "New York,NY,US", True),
        ("new york new york", "New York,NY,US", True),
        ("Kansas City, MO", "Kansas City,MO,US", True),
        ("Washington DC", "Washington,DC,US", True),
        ("St. Louis", "St. Louis", False),
        ("west virginia", "West Virginia", False),  # a state on its own is not "West", Virginia
        ("Paris, Ile-de-France", "Paris, Ile-de-France", True),  # unknown qualifier: passed through
    ],
)
def test_normalize_place(text, query, qualified):
    place = normalize_place(text)
    assert place["query"] == query and place["qualified"] is qualified


@pytest.mark.parametrize("name", ["Portland", "springfield", "Columbus", "Kansas City", "Birmingham"])
def test_ambiguous_bare_names(name):
    assert len(normalize_place(name)["ambiguous"]) >= 2


@pytest.mark.parametrize("text", ["Portland, Maine", "portland oregon", "Denver", "Paris", "London", "Austin"])
def test_not_ambiguous(text):
    assert normalize_place(text)["ambiguous"] == []


def test_ambiguity_question_and_answers():
    options = normalize_place("Springfield")["ambiguous"]
    assert ambiguity_question("Springfield", options) == (
        "There's more than one Springfield. Did you mean Springfield, Illinois, Springfield, Missouri, "
        "or Springfield, Massachusetts? Reply with the state or country."
    )
    assert pick_option("Missouri", options) == ("Springfield, Missouri", "Springfield,MO,US")
    assert pick_option("IL", options) == ("Springfield, Illinois", "Springfield,IL,US")
    assert pick_option("the one in massachusetts please", options)[1] == "Springfield,MA,US"
    assert pick_option("Illinois or Missouri", options) is None  # not exactly one
    assert pick_option("what about tomorrow", options) is None


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("Denver", "Denver,CO,US", True),
        ("denver", "Denver", True),
        ("Amarillo, Texas", "Amarillo,TX,US", True),
        ("Paris,FR", "Paris,TX,US", False),
        ("Paris", "Paris,FR", True),
        ("Denver", "Paris", False),
    ],
)
def test_same_place_query(a, b, same):
    assert same_place_query(a, b) is same


def _http(url, params, timeout):
    resp = MagicMock(status_code=200)
    city = params["q"].split(",")[0]
    resp.json.return_value = {"name": city, "sys": {"country": "US"},
                              "main": {"temp": 40.0, "feels_like": 35.0, "humidity": 50}, "weather": [{}]}
    return resp


def test_duplicate_lookups_in_one_turn_are_not_repeated(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-owm-key")

    class Model:
        def __init__(self):
            self.rounds = [
                [("get_current_weather", "Denver"), ("get_current_weather", "Denver, Colorado")],
                [("get_current_weather", "denver"), ("get_current_weather", "Paris")],
            ]

        def bind_tools(self, tools, **kwargs):
            model = self

            class _Bound:
                def invoke(self, messages):
                    if model.rounds:
                        calls = model.rounds.pop(0)
                        return AIMessage(content="", tool_calls=[
                            {"name": n, "args": {"city": c}, "id": f"{n}{i}{c}", "type": "tool_call"}
                            for i, (n, c) in enumerate(calls)])
                    return AIMessage(content="Done.")

            return _Bound()

    with patch.object(weather_tools.requests, "get", side_effect=_http) as get:
        result = run_turn(Model(), [], "What's the weather in Denver?")
    # The agent's Denver lookup plus Paris; the three repeated Denver calls reused the first result.
    assert [c.kwargs["params"]["q"] for c in get.call_args_list] == ["Denver", "Paris"]
    assert [c["args"]["city"] for c in result["tool_calls"]] == ["Denver", "Paris"]
    assert [c["result"]["location"] for c in result["shown_calls"]] == ["Denver, US"]
