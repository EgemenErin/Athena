import streamlit as st


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "llm_history": [],
        "df": None,
        "schema": None,
        "filename": None,
        "suggestions": None,
        "suggestions_for": None,
        "chart_suggestions": None,
        "chart_suggestions_for": None,
        "saved_charts": [],
        "cleaning_proposal": None,
        "df_original": None,
        "cleaning_applied": False,
        "active_tab": "Chat",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
