"""
Athena — universal CSV data chatbot powered by Ollama (qwen2.5-coder:7b).

Run with:
    streamlit run app.py
"""

import streamlit as st

from athena.ui import (
    init_session_state,
    inject_styles,
    render_cleaning_page,
    render_dashboard_page,
    render_main,
    render_sidebar,
)
from athena.ui.branding import inject_meta_tags

st.set_page_config(
    page_title="athena",
    page_icon="static/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
inject_meta_tags()
init_session_state()
render_sidebar()

section = st.radio(
    "Section",
    ["Chat", "Dashboard", "Clean data"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)

if section == "Chat":
    render_main()
elif section == "Dashboard":
    render_dashboard_page()
else:
    render_cleaning_page()
