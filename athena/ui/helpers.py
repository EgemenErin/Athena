import ollama
import pandas as pd
import streamlit as st

from athena.llm import build_schema_string, generate_suggested_questions


def add_to_llm_history(role: str, content: str, max_turns: int = 6) -> None:
    st.session_state.llm_history.append({"role": role, "content": content})
    if len(st.session_state.llm_history) > max_turns * 2:
        st.session_state.llm_history = st.session_state.llm_history[-(max_turns * 2) :]


def check_ollama() -> bool:
    try:
        models = [m["model"] for m in ollama.list()["models"]]
        return any("qwen2.5-coder" in m for m in models)
    except Exception:
        return False


def render_stat_cards(df: pd.DataFrame) -> None:
    null_pct = df.isna().sum().sum() / df.size * 100
    st.markdown(
        f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Rows</div>
                <div class="stat-value">{df.shape[0]:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Columns</div>
                <div class="stat-value">{df.shape[1]}</div>
            </div>
            <div class="stat-card wide">
                <div class="stat-label">Missing cells</div>
                <div class="stat-value">{null_pct:.1f}%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_suggestions(force: bool = False) -> list[str]:
    """Generate or return cached AI question suggestions for the current dataset."""
    fname = st.session_state.filename
    if not fname or st.session_state.df is None or st.session_state.schema is None:
        return []

    if (
        not force
        and st.session_state.suggestions
        and st.session_state.suggestions_for == fname
    ):
        return st.session_state.suggestions

    st.session_state.suggestions = generate_suggested_questions(
        st.session_state.df,
        st.session_state.schema,
        n=5,
    )
    st.session_state.suggestions_for = fname
    return st.session_state.suggestions


def load_csv_upload(uploaded) -> None:
    """Parse uploaded CSV into session state."""
    df = pd.read_csv(uploaded, low_memory=False)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    new_file = uploaded.name != st.session_state.get("_last_file")
    st.session_state.df = df
    st.session_state.schema = build_schema_string(df)
    st.session_state.filename = uploaded.name

    if new_file:
        st.session_state.messages = []
        st.session_state.llm_history = []
        st.session_state.suggestions = None
        st.session_state.suggestions_for = None
        st.session_state["_last_file"] = uploaded.name
