"""
Athena — universal CSV data chatbot powered by Ollama (qwen2.5-coder:7b).

Run with:
    streamlit run app.py
"""

import streamlit as st

from athena.ui import init_session_state, inject_styles, render_main, render_sidebar

st.set_page_config(
    page_title="Athena",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
init_session_state()
render_sidebar()
render_main()
