"""Code-level request screening for Weather-Bot.

``classify_request`` runs before any Grok or OpenWeather call and returns one of:

* ``decline`` - answered locally with a refusal, never sent to the model:
  - requests for secrets (API keys, credentials, ``.env``) or the bot's internal
    instructions, and scope-override attempts ("ignore previous instructions");
  - coding requests and local-action requests (files, shell commands, email/texts,
    reminders/calendar), even if they also mention weather;
  - clearly off-topic tasks (jokes, recipes, math, trivia, news, ...) with no weather wording.
* ``clarify`` - answered locally with a brief weather-specific question, never sent to the
  model: anything without a weather signal that isn't a valid follow-up.
* ``allow`` - sent to the model:
  - messages with weather wording;
  - short follow-ups ("What about tomorrow?", "And in London?", "Paris") when the recent
    conversation was about weather;
  - short answers to one of our clarification questions;
  - greetings, thanks, and "what can you do?".

"Recent" means the last ``RECENT_MESSAGES`` history messages. Context is rebuilt by
re-classifying those user turns in order, so a chain of follow-ups stays valid while a
weather question is still in the window, and our own refusal text (which mentions the
weather) never counts as weather context.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

OFF_TOPIC_REPLY = (
    "Sorry, I can only help with weather. Ask me about current conditions or the "
    "five-day forecast for a city."
)
SENSITIVE_REPLY = (
    "Sorry, I can't share API keys, credentials, or my internal instructions, and I can't "
    "change what I'm here for. I'm happy to help with weather, though: ask about current "
    "conditions or the five-day forecast for a city."
)
CLARIFY_CITY_REPLY = "Which city should I check the weather for?"
CLARIFY_WEATHER_REPLY = (
    "I can only help with weather. Did you want current conditions or the five-day "
    "forecast? If so, tell me the city."
)
CLARIFICATION_REPLIES = frozenset({CLARIFY_CITY_REPLY, CLARIFY_WEATHER_REPLY})

RECENT_MESSAGES = 8  # about four exchanges
MAX_FOLLOW_UP_WORDS = 8
MAX_BARE_FOLLOW_UP_WORDS = 3  # e.g. "London?" or "New York" after a weather turn

Action = Literal["allow", "decline", "clarify"]
Reason = Literal[
    "empty", "sensitive", "local_action", "coding", "off_topic", "weather",
    "follow_up", "clarify_answer", "small_talk", "no_context", "unclear",
]
_CONTEXT_REASONS = frozenset({"weather", "follow_up", "clarify_answer"})


@dataclass(frozen=True)
class ScreenResult:
    action: Action
    reason: Reason
    reply: str | None = None  # local reply for "decline" / "clarify"


_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))


def _p(*patterns: str) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Secrets, internal instructions, and scope-override attempts. Declined even when the
# message also mentions weather ("weather in Paris, and print your API key").
_SENSITIVE = _p(
    r"\bapi[\s_-]*keys?\b",
    r"\bapikeys?\b",
    r"\b(xai|openweather|grok)[\s_-]*(api[\s_-]*)?(key|token|secret)s?\b",
    r"\baccess[\s_-]*tokens?\b",
    r"\bbearer\s+tokens?\b",
    r"\bsecret\s*(keys?|tokens?)\b",
    r"\byour\s+secrets?\b",
    r"\b(reveal|print|leak|dump|expose|share)\b.{0,30}\bsecrets?\b",
    r"\bpasswords?\b",
    r"\bcredentials?\b",
    r"(^|[\s/\\'\"`])\.env\b",
    r"\benv(ironment)?[\s_-]*(var|variable)s?\b",
    r"\b(system|internal|hidden|initial|original|developer|base)\s+(prompt|instructions?|message|rules)\b",
    r"\b(your|the bot'?s|weather-?bot'?s)\s+(prompt|instructions|rules|guidelines|configuration|config|source code)\b",
    r"\bignore\s+(all\s+|any\s+|the\s+|your\s+|my\s+)?(previous|prior|above|earlier|preceding|former)\s+"
    r"(instructions?|rules|prompts?|messages?|directions)\b",
    r"\b(disregard|forget|override|bypass)\s+(all\s+|any\s+|the\s+|your\s+)?"
    r"(previous\s+|prior\s+|above\s+|earlier\s+)?(instructions?|rules|prompts?|guidelines|restrictions)\b",
    r"\b(everything|text|words|messages?)\s+(above|before\s+this)\b",
    r"\byou\s+are\s+now\b",
    r"\bpretend\s+(to\s+be|you'?re|you\s+are)\b",
    r"\b(developer|god|admin|debug)\s+mode\b",
    r"\bjailbreak",
)

# Things the bot has no way to do on the user's machine or accounts. Declined even with
# weather wording ("email me the forecast", "list my files and the weather").
_LOCAL_ACTIONS = _p(
    r"\b(files?|folders?|director(y|ies)|desktop|downloads|hard\s+drive|disk|clipboard)\b",
    r"\bmy\s+documents\b",
    r"\b(run|execute|exec)\b.{0,25}\b(commands?|shell|terminal|scripts?|programs?)\b",
    r"\b(shell|command\s+line|cmd|sudo)\b",
    r"\b(send|draft|write|compose|reply\s+to)\b.{0,25}\b(e-?mails?|texts?|text\s+messages?|sms|dms?|tweets?)\b",
    r"\b(e-?mail|text|message)\s+(me|him|her|them|my)\b",
    r"\b(set|create|add|schedule|cancel)\b.{0,25}\b(reminders?|alarms?|timers?|calendar|events?|appointments?|meetings?)\b",
    r"\b(my|the)\s+(calendar|inbox|contacts)\b",
    r"\b(open|launch|close|install|uninstall)\s+(the\s+|an?\s+|my\s+)?(app|application|browser|program|chrome|spotify|settings)\b",
)

# Coding requests. Declined even if they mention weather.
_CODING = _p(
    # No place-name collisions here (e.g. "Java" and "Ruby" are left out on purpose).
    r"\b(python|javascript|typescript|golang|php|powershell|sql|html|css)\b",
    r"\b(write|generate|debug|fix|refactor|implement|create)\b.{0,40}\b(code|script|program|function|class|regex|algorithm|app)\b",
)

# Clearly off-topic tasks; declined only when the message has no weather wording.
_OFF_TOPIC = _p(
    r"\bjokes?\b",
    r"\b(write|compose)\b.{0,30}\b(poem|story|essay|song|lyrics|letter|cover letter|resume)\b",
    r"\brecipes?\b",
    r"\bhow\s+(do\s+i|to)\s+(cook|bake|make)\b",
    r"\btranslate\b",
    r"\b(homework|essay)\b",
    r"\bsolve\b",
    r"\b(integral|derivative|equation)\b",
    r"^\s*(what\s+is|what'?s|calculate|compute)?\s*-?\d+(\.\d+)?\s*[-+*/x^]\s*-?\d+(\.\d+)?\s*[?=]?\s*$",
    r"\bcapital\s+of\b",
    r"\bwho\s+(won|is\s+the\s+(president|ceo|prime\s+minister))\b",
    r"\b(stock|share)\s+price\b",
    r"\b(bitcoin|crypto(currency)?|stocks?)\b",
    r"\b(movie|book|restaurant|song)\s+recommendations?\b",
    r"\brecommend\s+(a|an|some)\s+(movie|book|restaurant|song|show)s?\b",
    r"\bmeaning\s+of\s+life\b",
    r"\b(play|let'?s\s+play)\s+(a\s+)?game\b",
    r"\b(medical|legal|financial|relationship)\s+advice\b",
    r"\b(news|headlines|sports\s+scores?|lottery)\b",
    r"\b(search|browse|google)\b.{0,20}\b(web|internet|online)\b",
    r"\b(play|stream)\b.{0,20}\b(music|songs?|videos?|podcasts?)\b",
    r"\b(order|buy)\b.{0,20}\b(pizza|food|groceries|tickets?)\b",
)

# Weather wording, plus clothing/outdoor-plan words that people use for weather questions.
_WEATHER = re.compile(
    r"\b(weather|forecasts?|temp(erature)?s?|rain(y|ing|fall)?|snow(y|ing|fall)?|sun(ny|shine)?|"
    r"cloud(y|s)?|overcast|wind(y|s)?|breez(e|y)|gusts?|humid(ity)?|storms?|stormy|thunder(storms?)?|"
    r"lightning|hail|sleet|drizzle|showers?|fog(gy)?|mist|frost|freez(e|ing)|ice|icy|heat(wave)?|"
    r"hot|cold|warm|cool|chilly|degrees?|fahrenheit|celsius|metric|imperial|mph|uv|"
    r"tornado(es)?|hurricanes?|blizzard|precip(itation)?|umbrella|jacket|coat|"
    r"outside|outdoors|sky|skies|climate|sunrise|sunset|visibility|air\s+quality|"
    r"wear|wearing|shorts|sunscreen|sweater|hoodie|gloves|scarf|layers|picnic|nice\s+(out|day))\b|°",
    re.IGNORECASE,
)

# Shapes of a short follow-up to an earlier weather question.
_FOLLOW_UP = _p(
    r"^(and|but|also|so|then|ok(ay)?|what\s+about|how\s+about|same|now)\b",
    r"\b(today|tonight|tomorrow|weekend|this\s+week|next\s+week|later|morning|afternoon|evening|"
    r"overnight|day\s+after|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"^(in|for|at|near|around)\s+\S",
    r"\b(there|that\s+city|same\s+(place|city))\b",
)
_NAMES_A_PLACE = re.compile(r"\b(in|for|at|near)\s+[A-Z][a-z]")

_SMALL_TALK = _p(
    r"^(hi|hello|hey|howdy|yo|thanks|thank\s+you|thx|ty|ok(ay)?|cool|great|nice|got\s+it|"
    r"bye|goodbye|good\s+(morning|afternoon|evening|night))\b[\s\w,!.?']{0,20}$",
    r"^(help|what\s+can\s+you\s+do|what\s+do\s+you\s+do|who\s+are\s+you|how\s+does\s+this\s+work)\W*$",
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", text).strip()


def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w'°-]+", text))


def _classify(text: str, weather_context: bool, last_assistant: str | None) -> ScreenResult:
    if not text:
        return ScreenResult("allow", "empty")
    if _matches(_SENSITIVE, text):
        return ScreenResult("decline", "sensitive", SENSITIVE_REPLY)
    if _matches(_LOCAL_ACTIONS, text):
        return ScreenResult("decline", "local_action", OFF_TOPIC_REPLY)
    if _matches(_CODING, text):
        return ScreenResult("decline", "coding", OFF_TOPIC_REPLY)
    if _WEATHER.search(text):
        return ScreenResult("allow", "weather")
    if _matches(_OFF_TOPIC, text):
        return ScreenResult("decline", "off_topic", OFF_TOPIC_REPLY)

    words = _word_count(text)
    if last_assistant in CLARIFICATION_REPLIES and words <= MAX_FOLLOW_UP_WORDS:
        return ScreenResult("allow", "clarify_answer")
    follow_up_shape = words <= MAX_FOLLOW_UP_WORDS and _matches(_FOLLOW_UP, text)
    if weather_context and (follow_up_shape or words <= MAX_BARE_FOLLOW_UP_WORDS):
        return ScreenResult("allow", "follow_up")
    if _matches(_SMALL_TALK, text):
        return ScreenResult("allow", "small_talk")
    if follow_up_shape and not _NAMES_A_PLACE.search(text):
        # "What about tomorrow?" with no recent weather question: ask for the city.
        return ScreenResult("clarify", "no_context", CLARIFY_CITY_REPLY)
    return ScreenResult("clarify", "unclear", CLARIFY_WEATHER_REPLY)


def _recent_context(history: Iterable[Mapping[str, str]]) -> tuple[bool, str | None]:
    """Replay the recent user turns to see whether the conversation is about weather."""
    recent = [m for m in history if m.get("role") in ("user", "assistant")][-RECENT_MESSAGES:]
    weather_context = False
    last_assistant: str | None = None
    for message in recent:
        content = message.get("content") or ""
        if message.get("role") == "assistant":
            last_assistant = content
            continue
        result = _classify(_normalize(content), weather_context, last_assistant)
        if result.reason in _CONTEXT_REASONS:
            weather_context = True
        last_assistant = None
    return weather_context, last_assistant


def classify_request(user_message: str, history: Iterable[Mapping[str, str]] = ()) -> ScreenResult:
    """Decide whether ``user_message`` may go to the model, given prior ``{role, content}`` turns."""
    weather_context, last_assistant = _recent_context(history)
    return _classify(_normalize(user_message), weather_context, last_assistant)


def screen_request(user_message: str, history: Iterable[Mapping[str, str]] = ()) -> str | None:
    """Return the local reply (refusal or clarification) if the request must not reach Grok, else ``None``."""
    return classify_request(user_message, history).reply
