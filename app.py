"""
app.py
------
Universal CSV data chatbot powered by Ollama (qwen2.5-coder:7b).
Upload any CSV → ask questions in plain English → get tables, charts, insights.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import ollama

from llm_engine import (
    build_schema_string,
    generate_and_run,
    generate_suggested_questions,
    summarise_result,
)

# ── Page config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataChat",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
    --bg: #0c0d10;
    --surface: #14161c;
    --surface-2: #1a1d26;
    --border: #2a2f3d;
    --text: #e8eaef;
    --muted: #8b93a7;
    --accent: #e8a838;
    --accent-dim: #c4892a;
    --teal: #3dd6c6;
    --danger: #f07178;
}

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    color: var(--text);
}

.stApp {
    background: radial-gradient(ellipse 120% 80% at 50% -20%, #1a1520 0%, var(--bg) 55%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111318 0%, #0e1014 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}

.brand-title {
    font-family: 'Fraunces', serif;
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem 0;
    line-height: 1.1;
}

.brand-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 1rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(61, 214, 198, 0.08);
    border: 1px solid rgba(61, 214, 198, 0.25);
    color: var(--teal);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.status-pill::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 8px var(--teal);
}

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 0.75rem 0;
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
}

.stat-card.wide {
    grid-column: 1 / -1;
}

.stat-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 4px;
}

.stat-value {
    font-family: 'Fraunces', serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.1;
}

.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 1.25rem 0 0.6rem 0;
    font-weight: 600;
}

.suggest-hint {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    line-height: 1.4;
}

[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 10px;
    font-size: 0.92rem;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid var(--accent);
}

[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-dim) !important;
    box-shadow: 0 0 0 1px var(--accent-dim) !important;
}

code, pre {
    font-family: ui-monospace, 'Cascadia Code', monospace !important;
    font-size: 0.8rem !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

.insight-box {
    background: linear-gradient(135deg, rgba(232, 168, 56, 0.08) 0%, var(--surface) 100%);
    border: 1px solid rgba(232, 168, 56, 0.22);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 8px 0 12px;
    font-size: 0.92rem;
    line-height: 1.55;
    color: #f0e6d4;
}

.landing-hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    max-width: 640px;
    margin: 0 auto;
}

.landing-hero h1 {
    font-family: 'Fraunces', serif !important;
    font-size: 2.6rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
    letter-spacing: -0.04em;
    margin-bottom: 0.75rem !important;
}

.landing-hero p {
    color: var(--muted);
    font-size: 1.05rem;
    line-height: 1.6;
}

.step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.35rem;
    height: 100%;
}

.step-num {
    font-family: 'Fraunces', serif;
    color: var(--accent);
    font-size: 1.5rem;
    margin-bottom: 0.35rem;
}

.step-card h3 {
    font-size: 1rem !important;
    color: var(--text) !important;
    margin-bottom: 0.4rem !important;
}

.step-card p {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.5;
    margin: 0;
}

.chat-header {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.25rem 0;
}

.chat-meta {
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-dim) !important;
}

hr {
    border-color: var(--border) !important;
    margin: 1.25rem 0 !important;
}

/* Suggestion buttons */
[data-testid="stSidebar"] .stButton > button {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    padding: 0.65rem 0.85rem !important;
    font-size: 0.82rem !important;
    line-height: 1.35 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: border-color 0.15s, background 0.15s !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.5rem !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent-dim) !important;
    background: var(--surface-2) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(232, 168, 56, 0.12) !important;
    border-color: var(--accent-dim) !important;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state ────────────────────────────────────────────────────────────────

def _init_state():
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


_init_state()

# ── Helpers ──────────────────────────────────────────────────────────────────────

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#14161c",
    plot_bgcolor="#14161c",
    font_color="#e8eaef",
    font_family="Source Sans 3, sans-serif",
    colorway=["#e8a838", "#3dd6c6", "#f07178", "#a8b4ff", "#7ec97e"],
)


def try_make_chart(result):
    if not isinstance(result, (pd.DataFrame, pd.Series)):
        return None

    if isinstance(result, pd.Series):
        result = result.reset_index()
        result.columns = ["category", "value"]

    if result.shape[1] == 2:
        x_col, y_col = result.columns[0], result.columns[1]
        if pd.api.types.is_numeric_dtype(result[y_col]):
            fig = px.bar(
                result.head(20),
                x=x_col,
                y=y_col,
                color=y_col,
                color_continuous_scale=["#1a1d26", "#e8a838"],
            )
            fig.update_layout(
                **PLOTLY_THEME,
                xaxis_tickangle=-35,
                showlegend=False,
                margin=dict(t=30, b=60, l=40, r=20),
                coloraxis_showscale=False,
            )
            fig.update_traces(marker_line_width=0)
            return fig

    return None


def add_to_llm_history(role: str, content: str, max_turns: int = 6):
    st.session_state.llm_history.append({"role": role, "content": content})
    if len(st.session_state.llm_history) > max_turns * 2:
        st.session_state.llm_history = st.session_state.llm_history[-(max_turns * 2) :]


def check_ollama() -> bool:
    try:
        models = [m["model"] for m in ollama.list()["models"]]
        return any("qwen2.5-coder" in m for m in models)
    except Exception:
        return False


def render_stat_cards(df: pd.DataFrame):
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


def load_suggestions(force: bool = False):
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


# ── Sidebar ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="brand-title">DataChat</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-sub">Ask your CSV anything — runs locally via Ollama.</p>',
        unsafe_allow_html=True,
    )

    if check_ollama():
        st.markdown('<span class="status-pill">Local · qwen2.5-coder</span>', unsafe_allow_html=True)
    else:
        st.warning("Ollama not reachable or model missing.")

    st.markdown("---")

    uploaded = st.file_uploader(
        "Drop a CSV file",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
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

            st.success(f"Loaded **{uploaded.name}**")

        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown('<p class="section-label">Overview</p>', unsafe_allow_html=True)
        render_stat_cards(df)

        with st.expander("All columns", expanded=False):
            for col in df.columns:
                dtype_short = (
                    str(df[col].dtype)
                    .replace("object", "str")
                    .replace("int64", "int")
                    .replace("float64", "float")
                )
                st.caption(f"`{col}` · {dtype_short}")

        st.markdown('<p class="section-label">Suggested questions</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="suggest-hint">Tailored to your columns — click to ask, or refresh for new ideas.</p>',
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns([1, 1])
        with col_b:
            refresh = st.button("↻ New ideas", use_container_width=True, key="refresh_suggestions")

        if refresh:
            st.session_state.suggestions = None
            st.session_state.suggestions_for = None

        need_generate = (
            st.session_state.suggestions is None
            or st.session_state.suggestions_for != st.session_state.filename
        )

        if need_generate:
            with st.spinner("Generating ideas for this dataset…"):
                suggestions = load_suggestions(force=True)
        else:
            suggestions = st.session_state.suggestions or []

        for i, s in enumerate(suggestions):
            if st.button(s, key=f"sug_{i}_{hash(s) % 10**6}", use_container_width=True):
                st.session_state["_prefill"] = s

        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.llm_history = []
            st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────────

if st.session_state.df is None:
    st.markdown(
        """
        <div class="landing-hero">
            <h1>Talk to your data</h1>
            <p>Upload any CSV and ask questions in plain English.
            DataChat writes the pandas code, runs it locally, and explains what it found.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">1</div>
                <h3>Upload</h3>
                <p>Drop a CSV in the sidebar — surveys, sales, logs, exports, anything tabular.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">2</div>
                <h3>Ask</h3>
                <p>Type a question or pick an AI-suggested prompt matched to your columns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-num">3</div>
                <h3>Explore</h3>
                <p>Get tables, charts, and a short narrative insight for every answer.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Runs on your machine · Ollama · your data never leaves this device")

else:
    df = st.session_state.df
    st.markdown(f'<p class="chat-header">{st.session_state.filename}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="chat-meta">{df.shape[0]:,} rows · {df.shape[1]} columns</p>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                if msg.get("narrative"):
                    st.markdown(
                        f'<div class="insight-box">{msg["narrative"]}</div>',
                        unsafe_allow_html=True,
                    )
                if msg.get("result_df") is not None:
                    st.dataframe(msg["result_df"], use_container_width=True)
                if msg.get("chart") is not None:
                    st.plotly_chart(msg["chart"], use_container_width=True)
                if msg.get("scalar") is not None:
                    st.metric("Result", msg["scalar"])
                if msg.get("error"):
                    st.error(msg["error"])
                if msg.get("code"):
                    with st.expander("View generated code"):
                        st.code(msg["code"], language="python")

    prefill = st.session_state.pop("_prefill", None)
    question = st.chat_input("Ask anything about your data…") or prefill

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                output = generate_and_run(
                    question=question,
                    df=st.session_state.df,
                    schema=st.session_state.schema,
                    chat_history=st.session_state.llm_history,
                )

            result = output["result"]
            error = output["error"]
            code = output["code"]

            msg_data = {
                "role": "assistant",
                "code": code,
                "error": None,
                "result_df": None,
                "chart": None,
                "scalar": None,
                "narrative": None,
            }

            if error:
                st.error(error)
                msg_data["error"] = error
            else:
                narrative = summarise_result(question, result)
                st.markdown(
                    f'<div class="insight-box">{narrative}</div>',
                    unsafe_allow_html=True,
                )
                msg_data["narrative"] = narrative

                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                    msg_data["result_df"] = result
                    chart = try_make_chart(result)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        msg_data["chart"] = chart

                elif isinstance(result, pd.Series):
                    df_result = result.reset_index()
                    st.dataframe(df_result, use_container_width=True)
                    msg_data["result_df"] = df_result
                    chart = try_make_chart(result)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                        msg_data["chart"] = chart

                else:
                    st.metric("Result", result)
                    msg_data["scalar"] = result

                if code:
                    with st.expander("View generated code"):
                        st.code(code, language="python")

                add_to_llm_history("user", question)
                add_to_llm_history("assistant", f"```python\n{code}\n```")

            st.session_state.messages.append(msg_data)
