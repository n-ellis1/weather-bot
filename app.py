"""Weather-Bot: Streamlit chat page backed by Grok and the OpenWeather tools."""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from chat_agent import build_model, run_turn

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="Weather-Bot", page_icon="🌦️")
st.title("🌦️ Weather-Bot")
st.caption("Ask about current weather or the five-day forecast for any city.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


@st.cache_resource
def get_model():
    return build_model()


def answer(prompt: str, history: list[dict[str, str]]) -> str:
    if not os.getenv("XAI_API_KEY", "").strip():
        return "Weather-Bot isn't configured yet: the Grok API key (`XAI_API_KEY`) is missing."
    try:
        return run_turn(get_model(), history, prompt)["reply"]
    except Exception as exc:  # noqa: BLE001 - show the error type only; raw text may include request details
        return f"Sorry, the assistant couldn't complete that request ({type(exc).__name__}). Please try again."


if prompt := st.chat_input("Ask about the weather..."):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the weather..."):
            reply = answer(prompt, history)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
