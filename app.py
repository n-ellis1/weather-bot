"""Weather-Bot: minimal Streamlit chat page (placeholder responses until Grok is wired in)."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Weather-Bot", page_icon="🌦️")
st.title("🌦️ Weather-Bot")
st.caption("Ask about current weather or the five-day forecast for any city.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the weather..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder until the Grok agent and OpenWeather tools are implemented.
    reply = "Weather-Bot is set up, but weather lookups aren't connected yet."
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
