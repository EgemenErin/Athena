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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
