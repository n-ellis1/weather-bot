"""Grok-powered chat flow for Weather-Bot.

A small, explicit tool-calling loop around ``ChatXAI``:

* Only ``get_current_weather`` and ``get_weather_forecast`` are bound, and the
  executor refuses any other tool name the model might emit.
* Each user turn may run at most ``MAX_TOOL_CALLS_PER_TURN`` tool calls across at
  most ``MAX_TOOL_ROUNDS`` model rounds. After that the model is asked for a final
  answer with tools disabled, so the loop always terminates.
* Tool results (including ``{"ok": False, "error": ...}`` failures) are passed back
  to the model verbatim as JSON so answers are grounded in them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from weather_tools import get_current_weather, get_weather_forecast

MODEL_NAME = "grok-4.7"
MAX_TOOL_CALLS_PER_TURN = 4
MAX_TOOL_ROUNDS = 3
MODEL_TIMEOUT_SECONDS = 60

WEATHER_TOOLS: dict[str, BaseTool] = {
    get_current_weather.name: get_current_weather,
    get_weather_forecast.name: get_weather_forecast,
}

SYSTEM_PROMPT = """You are Weather-Bot, an assistant that answers weather questions.

Tools:
- get_current_weather: conditions right now for a city.
- get_weather_forecast: the 5-day forecast for a city, summarized per local day.

Rules:
- For any question about live, current, today's, upcoming, or forecast weather, call the
  matching tool. Never answer those from memory.
- Base every weather fact (temperatures, conditions, humidity, wind, dates, location) only on
  tool results from this conversation. Do not invent or estimate values the tools did not return.
- Use the unit labels returned by the tool. Pass units="imperial" when the user asks for
  Fahrenheit or mph, otherwise use the default.
- Mention the resolved location returned by the tool. If a forecast day is marked partial,
  say that it only covers part of the day.
- If a tool result has "ok": false, tell the user plainly that the lookup failed and quote its
  "error" message. Do not guess or fill in weather data for that request.
- Keep replies short and clear."""

TOOL_LIMIT_MESSAGE = (
    "Tool call limit reached for this message; this call was not run. "
    "Answer using only the tool results already returned."
)
NO_ANSWER_FALLBACK = "Sorry, I couldn't produce an answer for that. Please try rephrasing your question."


def build_model(model_name: str = MODEL_NAME) -> BaseChatModel:
    """Create the ChatXAI client. The API key is read from ``XAI_API_KEY`` by ChatXAI itself."""
    from langchain_xai import ChatXAI

    return ChatXAI(model=model_name, temperature=0, timeout=MODEL_TIMEOUT_SECONDS, max_retries=2)


def to_langchain_history(messages: Iterable[Mapping[str, str]]) -> list[BaseMessage]:
    """Convert ``st.session_state.messages`` dicts into LangChain messages."""
    history: list[BaseMessage] = []
    for message in messages:
        role, content = message.get("role"), message.get("content", "")
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def _run_tool(name: str, args: Any) -> dict[str, Any]:
    tool = WEATHER_TOOLS.get(name)
    if tool is None:
        return {"ok": False, "error": f"Unknown tool '{name}'. Only weather tools are available."}
    if not isinstance(args, dict):
        return {"ok": False, "error": "The tool was called with invalid arguments."}
    try:
        result = tool.invoke(args)
    except Exception:  # noqa: BLE001 - e.g. argument validation errors; never surface raw text
        return {"ok": False, "error": "The tool was called with invalid arguments."}
    if not isinstance(result, dict):
        return {"ok": False, "error": "The weather tool returned an unexpected result."}
    return result


def _text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content or []]
    return "".join(parts).strip()


def run_turn(
    model: BaseChatModel,
    history: Iterable[Mapping[str, str]],
    user_message: str,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> dict[str, Any]:
    """Answer one user message.

    ``history`` holds prior ``{"role", "content"}`` turns (not including ``user_message``).
    Returns ``{"reply": str, "tool_calls": [{"name", "args", "result"}, ...]}``.
    """
    tools = list(WEATHER_TOOLS.values())
    with_tools = model.bind_tools(tools)
    no_tools = model.bind_tools(tools, tool_choice="none")

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        *to_langchain_history(history),
        HumanMessage(content=user_message),
    ]
    executed: list[dict[str, Any]] = []

    for _ in range(max_rounds):
        response = with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return {"reply": _text(response) or NO_ANSWER_FALLBACK, "tool_calls": executed}

        for call in response.tool_calls:
            if len(executed) >= max_tool_calls:
                result = {"ok": False, "error": TOOL_LIMIT_MESSAGE}
            else:
                result = _run_tool(call["name"], call.get("args"))
                executed.append({"name": call["name"], "args": call.get("args"), "result": result})
            messages.append(
                ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"], name=call["name"])
            )

    # Round budget exhausted: ask for a final answer with tools disabled.
    final = no_tools.invoke(messages)
    reply = _text(final)
    if not reply:
        errors = [c["result"]["error"] for c in executed if not c["result"].get("ok") and c["result"].get("error")]
        reply = "The weather lookup failed: " + " ".join(errors) if errors else NO_ANSWER_FALLBACK
    return {"reply": reply, "tool_calls": executed}
