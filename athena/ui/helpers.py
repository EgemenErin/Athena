import ollama
import pandas as pd
import streamlit as st

from athena.llm import build_schema_string, generate_chart_suggestions, generate_suggested_questions
from athena.llm.schema import coerce_numeric_like_columns


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


def cleaned_download_filename() -> str:
    raw = st.session_state.filename or "dataset.csv"
    if raw.lower().endswith(".csv"):
        stem = raw[:-4]
    else:
        stem = raw
    suffix = "_cleaned" if st.session_state.cleaning_applied else "_export"
    return f"{stem}{suffix}.csv"


def render_download_csv_button(*, type_primary: bool = False) -> None:
    df = st.session_state.df
    if df is None:
        return
    label = (
        "Download cleaned CSV"
        if st.session_state.cleaning_applied
        else "Download CSV"
    )
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=cleaned_download_filename(),
        mime="text/csv",
        type="primary" if type_primary else "secondary",
        use_container_width=True,
        key="download_cleaned_csv",
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


def load_chart_suggestions(force: bool = False) -> list[dict]:
    """Generate or return cached chart suggestions for the current dataset."""
    fname = st.session_state.filename
    if not fname or st.session_state.df is None or st.session_state.schema is None:
        return []

    if (
        not force
        and st.session_state.chart_suggestions
        and st.session_state.chart_suggestions_for == fname
    ):
        return st.session_state.chart_suggestions

    st.session_state.chart_suggestions = generate_chart_suggestions(
        st.session_state.df,
        st.session_state.schema,
        n=6,
    )
    st.session_state.chart_suggestions_for = fname
    return st.session_state.chart_suggestions


def load_csv_upload(uploaded) -> bool:
    """
    Parse uploaded CSV into session state.
    Returns True if this was a new file load (False when the uploader still
    holds the same file on rerun — keeps cleaned/edited in-memory df intact).
    """
    new_file = uploaded.name != st.session_state.get("_last_file")
    if not new_file:
        return False

    df = pd.read_csv(uploaded, low_memory=False)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    df = coerce_numeric_like_columns(df)
    st.session_state.df = df
    st.session_state.schema = build_schema_string(df)
    st.session_state.filename = uploaded.name
    st.session_state.messages = []
    st.session_state.llm_history = []
    st.session_state.suggestions = None
    st.session_state.suggestions_for = None
    st.session_state.chart_suggestions = None
    st.session_state.chart_suggestions_for = None
    st.session_state.saved_charts = []
    st.session_state.pop("chart_builder_preview", None)
    st.session_state.cleaning_proposal = None
    st.session_state.df_original = None
    st.session_state.cleaning_applied = False
    st.session_state["_last_file"] = uploaded.name
    return True
