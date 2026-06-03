import html
import uuid
from typing import Any

import pandas as pd
import streamlit as st

from athena.llm.chart_suggestions import _spec_key, chart_description, suggest_charts_for_columns
from athena.ui.charts import build_chart_from_spec
from athena.ui.helpers import load_chart_suggestions, render_stat_cards
from athena.ui.pdf_export import build_dashboard_pdf


def _saved_spec_keys() -> set[tuple]:
    return {_spec_key(s) for s in st.session_state.saved_charts}


def _append_saved_chart(spec: dict[str, Any]) -> None:
    key = _spec_key(spec)
    if key in _saved_spec_keys():
        return
    entry = dict(spec)
    entry["saved_id"] = str(uuid.uuid4())
    st.session_state.saved_charts.append(entry)


def _remove_saved_chart(saved_id: str) -> None:
    st.session_state.saved_charts = [
        s for s in st.session_state.saved_charts if s.get("saved_id") != saved_id
    ]


def _pdf_filename() -> str:
    raw = st.session_state.filename or "dataset.csv"
    stem = raw[:-4] if raw.lower().endswith(".csv") else raw
    return f"{stem}_charts.pdf"


def _render_chart_card(
    df: pd.DataFrame,
    spec: dict[str, Any],
    *,
    key_prefix: str,
    show_save: bool = False,
    show_remove: bool = False,
    saved_id: str | None = None,
) -> None:
    st.markdown(f"**{spec.get('title', 'Chart')}**")
    desc = chart_description(spec)
    if desc:
        safe = html.escape(desc)
        st.markdown(f'<p class="chart-desc">{safe}</p>', unsafe_allow_html=True)
    fig = build_chart_from_spec(df, spec)
    if fig is None:
        st.warning("Could not render this chart.")
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_plot")

    btn_cols = st.columns(2) if (show_save or show_remove) else []
    col_idx = 0
    if show_save:
        with btn_cols[col_idx]:
            spec_key = _spec_key(spec)
            already = spec_key in _saved_spec_keys()
            if st.button(
                "Saved" if already else "Save",
                key=f"{key_prefix}_save",
                disabled=already,
                use_container_width=True,
            ):
                _append_saved_chart(spec)
                st.rerun()
        col_idx += 1
    if show_remove and saved_id:
        with btn_cols[col_idx]:
            if st.button("Remove", key=f"{key_prefix}_remove", use_container_width=True):
                _remove_saved_chart(saved_id)
                st.rerun()


def _render_chart_grid(
    df: pd.DataFrame,
    specs: list[dict[str, Any]],
    *,
    key_prefix: str,
    show_save: bool = False,
    show_remove: bool = False,
) -> None:
    for row_start in range(0, len(specs), 2):
        cols = st.columns(2)
        for col_idx, spec in enumerate(specs[row_start : row_start + 2]):
            with cols[col_idx]:
                sid = spec.get("saved_id") if show_remove else None
                _render_chart_card(
                    df,
                    spec,
                    key_prefix=f"{key_prefix}_{row_start}_{col_idx}",
                    show_save=show_save,
                    show_remove=show_remove,
                    saved_id=sid,
                )


@st.dialog("Make a chart")
def _chart_builder_dialog(df: pd.DataFrame) -> None:
    columns = list(df.columns)
    x_col = st.selectbox("X column", columns, key="chart_builder_x")
    y_options = ["— none —", *columns]
    y_pick = st.selectbox("Y column (optional)", y_options, key="chart_builder_y")
    y_col = None if y_pick == "— none —" else y_pick

    options = suggest_charts_for_columns(df, x_col, y_col)
    if not options:
        st.warning("No chart types fit these columns. Try different columns.")
        return

    labels = [
        f"{s['chart_type'].title()} — {s.get('title', 'Chart')}" for s in options
    ]
    choice = st.radio("Suggested chart", labels, key="chart_builder_choice")
    chosen = options[labels.index(choice)]
    st.markdown(
        f'<p class="chart-desc">{html.escape(chart_description(chosen))}</p>',
        unsafe_allow_html=True,
    )

    custom_title = st.text_input(
        "Chart title (optional)",
        value=chosen.get("title", ""),
        key="chart_builder_title",
    )

    preview_spec = dict(chosen)
    if custom_title.strip():
        preview_spec["title"] = custom_title.strip()

    if st.button("OK", type="primary", key="chart_builder_ok"):
        st.session_state.chart_builder_preview = preview_spec

    preview = st.session_state.get("chart_builder_preview")
    if preview:
        st.markdown("**Preview**")
        fig = build_chart_from_spec(df, preview)
        if fig is None:
            st.warning("Could not render this chart.")
        else:
            st.plotly_chart(fig, use_container_width=True, key="chart_builder_preview_plot")

        if st.button("Save chart", type="primary", key="chart_builder_save"):
            _append_saved_chart(preview)
            st.session_state.chart_builder_preview = None
            st.rerun()


def render_dashboard_page() -> None:
    st.markdown(
        '<p class="chat-header">Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="chat-meta">Suggested graphs for your dataset — build, save, and export your own charts.</p>',
        unsafe_allow_html=True,
    )

    df = st.session_state.df
    if df is None:
        st.markdown(
            """
            <div class="landing-hero">
                <h1>Visualize your CSV</h1>
                <p>Upload a file in the sidebar to see AI-suggested charts
                tailored to your columns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"Working on {st.session_state.filename}")
    render_stat_cards(df)
    st.markdown("---")

    tool_a, tool_b, tool_c = st.columns([1, 1, 1])
    with tool_a:
        if st.button("+ Make a chart", use_container_width=True, key="open_chart_builder"):
            _chart_builder_dialog(df)
    with tool_b:
        refresh = st.button(
            "↻ Refresh suggestions",
            use_container_width=True,
            key="refresh_chart_suggestions",
        )
    with tool_c:
        saved = st.session_state.saved_charts
        pdf_bytes = None
        pdf_error = None
        if saved:
            try:
                pdf_bytes = build_dashboard_pdf(df, saved)
            except Exception as exc:
                pdf_error = str(exc)
        st.download_button(
            "Download PDF",
            data=pdf_bytes or b"",
            file_name=_pdf_filename(),
            mime="application/pdf",
            use_container_width=True,
            disabled=not saved or pdf_bytes is None,
            key="download_charts_pdf",
        )
    if pdf_error:
        st.error(f"PDF export failed: {pdf_error}. Try: pip install kaleido")

    if refresh:
        st.session_state.chart_suggestions = None
        st.session_state.chart_suggestions_for = None

    saved_charts = st.session_state.saved_charts
    if saved_charts:
        st.markdown(
            '<p class="section-label">My charts</p>',
            unsafe_allow_html=True,
        )
        _render_chart_grid(
            df,
            saved_charts,
            key_prefix="saved",
            show_remove=True,
        )
        st.markdown("---")

    st.markdown(
        '<p class="section-label">Suggested graphs</p>',
        unsafe_allow_html=True,
    )

    need_generate = (
        st.session_state.chart_suggestions is None
        or st.session_state.chart_suggestions_for != st.session_state.filename
    )

    if need_generate:
        with st.spinner("Generating chart ideas…"):
            specs = load_chart_suggestions(force=True)
    else:
        specs = st.session_state.chart_suggestions or []

    if not specs and not saved_charts:
        st.info("No chart suggestions available for this dataset.")
        return

    if specs:
        _render_chart_grid(
            df,
            specs,
            key_prefix="suggested",
            show_save=True,
        )
