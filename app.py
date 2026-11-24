"""Weather-Bot: Streamlit chat page backed by Grok and the OpenWeather tools."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from chat_agent import build_model, lookups_from, run_turn
from presentation import (
    DAY_CARDS_CSS,
    EXAMPLE_QUESTIONS,
    PARTIAL_NOTE,
    PRECIP_NOTE,
    UNAVAILABLE,
    UNIT_LABELS,
    build_views,
    day_cards_html,
    details_rows,
    forecast_title,
)
from safeguards import screen_request
from weather_tools import DEFAULT_UNITS

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="Weather-Bot", page_icon="🌦️", layout="wide")
st.markdown(
    """
    <style>
    /* Wider than the default centered column, but capped so lines stay readable. */
    [data-testid="stMainBlockContainer"] { max-width: 1100px; padding-top: 2.5rem; padding-bottom: 6rem; }
    [data-testid="stChatMessage"] { padding: 1rem 1.25rem; margin-bottom: 1rem; gap: 1rem; line-height: 1.6; }
    [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] { margin-top: 0.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.75rem; }
    /* Phones: give the cards the width instead of the chat bubble padding. */
    @media (max-width: 640px) {
      [data-testid="stMainBlockContainer"] { padding-left: 0.75rem; padding-right: 0.75rem; }
      [data-testid="stChatMessage"] { padding: 0.75rem 0.5rem; gap: 0.5rem; }
      [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] { padding: 0.75rem 0.6rem; }
    }
    """
    + DAY_CARDS_CSS
    + """
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "units" not in st.session_state:
    st.session_state.units = DEFAULT_UNITS

# --- Sidebar ------------------------------------------------------------------------------------
clicked_example = None
with st.sidebar:
    st.header("Settings")
    st.radio(
        "Units",
        options=["imperial", "metric"],
        format_func=lambda u: UNIT_LABELS[u]["name"],
        key="units",
        help="Used for new replies. Asking for a unit in chat (e.g. “in Celsius”) overrides it for that reply.",
    )
    st.divider()
    st.subheader("Try asking")
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(question, key=f"example_{i}", width="stretch"):
            clicked_example = question
    st.divider()
    if st.button("Clear chat", key="clear_chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()


# --- Rendering ----------------------------------------------------------------------------------
def render_current(view: dict) -> None:
    with st.container(border=True):
        st.markdown(f"#### 📍 {view['location']}")
        st.markdown(f"{view.get('icon', '')} **{view['description']}**".strip())
        metrics = view["metrics"]
        if metrics:
            for col, metric in zip(st.columns(len(metrics)), metrics):
                col.metric(metric["label"], metric["value"])
        if view.get("observed"):
            st.caption(view["observed"])


def render_forecast(view: dict) -> None:
    single = view["layout"] == "single"
    with st.container(border=True):
        st.markdown(f"#### 🗓️ {forecast_title(view)}")
        if not single:
            st.markdown(view["outlook"])
        st.markdown(day_cards_html(view["days"], single=single), unsafe_allow_html=True)
        if not single:
            with st.expander("Daily details: humidity and wind"):
                st.table(pd.DataFrame(details_rows(view["days"])).set_index("Day"))
        notes = []
        if view["day_not_in_window"]:
            notes.append("The day you asked about is outside the five-day window, so all available days are shown.")
        notes.append(PRECIP_NOTE)
        if view["has_partial"]:
            notes.append(PARTIAL_NOTE)
        if any(UNAVAILABLE in (d["precip"], d["humidity"], d["wind"], d["high"], d["low"]) for d in view["days"]):
            notes.append(f"{UNAVAILABLE} means the weather service didn't provide that value.")
        st.caption(" ".join(notes))


def render_legacy_table(view: dict) -> None:
    """Forecast views stored by the earlier table-only UI (still in an open session)."""
    with st.container(border=True):
        st.markdown(f"#### 🗓️ Forecast · {view['location']}")
        st.table(pd.DataFrame(view["rows"], columns=view["columns"]).set_index("Day"))


def render_message(message: dict) -> None:
    if message.get("lookup_failed"):
        st.warning(message["content"], icon="⚠️")
    else:
        st.markdown(message["content"])
    for view in message.get("views", []):
        if view["kind"] == "current":
            render_current(view)
        elif "days" in view:
            render_forecast(view)
        else:
            render_legacy_table(view)
    if message.get("views"):
        units = UNIT_LABELS[message["units"]]
        st.caption(f"Units: {units['temperature']}, {units['wind_speed']}")


@st.cache_resource
def get_model():
    return build_model()


def answer(prompt: str, history: list[dict], units: str) -> dict:
    """Return the assistant message dict for ``prompt``: ``content``, plus ``units``/``views`` for weather replies."""
    # Decline or clarify non-weather requests locally, before building or calling the model.
    if (local_reply := screen_request(prompt, history)) is not None:
        return {"role": "assistant", "content": local_reply}
    if not os.getenv("XAI_API_KEY", "").strip():
        return {
            "role": "assistant",
            "content": "Weather-Bot isn't configured yet: the Grok API key (`XAI_API_KEY`) is missing.",
        }
    try:
        result = run_turn(get_model(), history, prompt, units=units)
    except Exception as exc:  # noqa: BLE001 - show the error type only; raw text may include request details
        return {
            "role": "assistant",
            "content": f"Sorry, the assistant couldn't complete that request ({type(exc).__name__}). Please try again.",
        }
    message = {"role": "assistant", "content": result["reply"]}
    if result.get("lookup_failed"):
        message["lookup_failed"] = True  # rendered as a warning, not as a normal answer
    # Only lookups for the requested place get cards (never a wrong-city result).
    shown = result.get("shown_calls", result.get("tool_calls", []))
    views = build_views(prompt, shown)
    if views:
        message["units"] = views[0]["units"]
        message["views"] = views
    # Which cities resolved this turn, so a later follow-up can fetch fresh data for them.
    if lookups := lookups_from(shown):
        message["lookups"] = lookups
    if result.get("ambiguous_place"):  # "Which Portland?": the next answer picks one of these options
        message["ambiguous_place"] = result["ambiguous_place"]
    return message


# --- Page ---------------------------------------------------------------------------------------
st.title("🌦️ Weather-Bot")
st.caption("Current conditions and five-day forecasts for any city.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message)

if prompt := (st.chat_input("Ask about the weather...") or clicked_example):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the weather..."):
            reply = answer(prompt, history, st.session_state.units)
        render_message(reply)
    st.session_state.messages.append(reply)
