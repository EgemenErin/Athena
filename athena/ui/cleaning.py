import pandas as pd
import streamlit as st

from athena.llm import build_schema_string
from athena.llm.cleaning import (
    SUPPORTED_TYPES,
    analyze_for_cleaning,
    apply_cleaning_actions,
    outlier_mask,
    preview_stats,
    validate_action,
)
from athena.llm.schema import coerce_numeric_like_columns, numeric_columns
from athena.ui.helpers import render_download_csv_button, render_stat_cards


def _clear_cleaning_checkbox_keys() -> None:
    for key in list(st.session_state.keys()):
        if (
            key.startswith("clean_")
            or key.startswith("disabled_clean_")
            or key.startswith("clean_resolve_")
        ):
            del st.session_state[key]


def _action_label(action: dict) -> str:
    action_type = action.get("type", "unknown")
    col = action.get("column", "")
    reason = action.get("reason", "")
    if action_type == "drop_column":
        head = f"Drop column `{col}`"
    elif action_type == "fill_all_nulls":
        head = "Fill all missing values (0% null cells)"
    elif action_type == "fill_null":
        strategy = action.get("strategy", "median")
        head = f"Fill nulls in `{col}` ({strategy})"
    elif action_type == "drop_duplicate_rows":
        head = "Remove duplicate rows"
    elif action_type == "drop_rows_all_null":
        cols = action.get("columns")
        if cols:
            head = f"Drop rows where all null in {cols}"
        else:
            head = "Drop rows that are entirely empty"
    elif action_type == "drop_outlier_rows":
        method = action.get("method", "iqr")
        head = f"Remove outlier rows in `{col}` ({method})"
    elif action_type == "cap_outliers":
        lo = action.get("lower_percentile", 1)
        hi = action.get("upper_percentile", 99)
        head = f"Cap extreme values in `{col}` (P{lo}–P{hi})"
    elif action_type == "skip":
        head = f"No change for `{col}`"
    else:
        head = action_type
    if reason:
        return f"{head} — {reason}"
    return head


def _resolution_choices(action: dict, df: pd.DataFrame) -> list[tuple[str, dict]]:
    action_type = action.get("type")
    col = action.get("column")
    if not col or col not in df.columns:
        return [(_action_label(action), action)]

    reason = action.get("reason", "")
    aid = action.get("id")
    n = len(df)
    nulls = int(df[col].isna().sum())
    base = {"column": col, "reason": reason, "id": aid}

    def _pack(label: str, payload: dict) -> tuple[str, dict]:
        return label, {**base, **payload, "id": aid}

    if action_type == "drop_column":
        choices = [_pack("Drop column", {"type": "drop_column"})]
        if nulls > 0 and (n == 0 or nulls / n < 0.9):
            if col in numeric_columns(df):
                for strategy in ("median", "mean", "mode"):
                    choices.append(
                        _pack(f"Fill nulls ({strategy})", {"type": "fill_null", "strategy": strategy})
                    )
            else:
                choices.append(_pack("Fill nulls (mode)", {"type": "fill_null", "strategy": "mode"}))
        return choices

    if action_type == "fill_null":
        strategy = action.get("strategy", "median")
        choices = [_pack(f"Fill nulls ({strategy})", {"type": "fill_null", "strategy": strategy})]
        if col in numeric_columns(df):
            for s in ("median", "mean", "mode"):
                if s != strategy:
                    choices.append(_pack(f"Fill nulls ({s})", {"type": "fill_null", "strategy": s}))
        elif strategy != "mode":
            choices.append(_pack("Fill nulls (mode)", {"type": "fill_null", "strategy": "mode"}))
        choices.append(_pack("Drop column", {"type": "drop_column"}))
        return choices

    if action_type in ("drop_outlier_rows", "cap_outliers") and col in numeric_columns(df):
        lo = action.get("lower_percentile", 1)
        hi = action.get("upper_percentile", 99)
        drop_iqr = _pack(
            "Remove outlier rows (IQR)",
            {"type": "drop_outlier_rows", "method": "iqr", "factor": 1.5},
        )
        drop_z = _pack(
            "Remove outlier rows (z-score)",
            {"type": "drop_outlier_rows", "method": "zscore", "threshold": 3.0},
        )
        cap = _pack(
            f"Cap extreme values (P{lo}–P{hi})",
            {"type": "cap_outliers", "lower_percentile": lo, "upper_percentile": hi},
        )
        if action_type == "cap_outliers":
            return [cap, drop_iqr, drop_z]
        if action.get("method") == "zscore":
            return [drop_z, drop_iqr, cap]
        return [drop_iqr, drop_z, cap]

    return [(_action_label(action), action)]


def _resolved_action(action: dict, df: pd.DataFrame) -> dict:
    aid = action.get("id", action.get("type"))
    choices = _resolution_choices(action, df)
    if len(choices) == 1:
        return choices[0][1]

    labels = [c[0] for c in choices]
    resolve_key = f"clean_resolve_{aid}"
    picked = st.session_state.get(resolve_key, labels[0])
    if picked not in labels:
        picked = labels[0]
    for label, resolved in choices:
        if label == picked:
            return resolved
    return choices[0][1]


def _selected_actions(proposal: dict, df: pd.DataFrame) -> list[dict]:
    actions = proposal.get("actions", [])
    selected = []
    for action in actions:
        if action.get("type") == "skip":
            continue
        key = f"clean_{action.get('id', action.get('type'))}"
        if st.session_state.get(key, True):
            selected.append(_resolved_action(action, df))
    return selected


def _apply_cleaning(selected: list[dict]) -> None:
    if st.session_state.df_original is None:
        st.session_state.df_original = st.session_state.df.copy()

    st.session_state.df = coerce_numeric_like_columns(
        apply_cleaning_actions(st.session_state.df, selected)
    )
    st.session_state.schema = build_schema_string(st.session_state.df)
    st.session_state.cleaning_proposal = None
    st.session_state.cleaning_applied = True
    st.session_state.messages = []
    st.session_state.llm_history = []
    st.session_state.suggestions = None
    st.session_state.suggestions_for = None
    st.session_state.chart_suggestions = None
    st.session_state.chart_suggestions_for = None
    st.rerun()


def _restore_original() -> None:
    if st.session_state.df_original is not None:
        st.session_state.df = st.session_state.df_original.copy()
        st.session_state.schema = build_schema_string(st.session_state.df)
        st.session_state.df_original = None
        st.session_state.cleaning_applied = False
        st.session_state.cleaning_proposal = None
        st.session_state.messages = []
        st.session_state.llm_history = []
        st.session_state.suggestions = None
        st.session_state.suggestions_for = None
        st.session_state.chart_suggestions = None
        st.session_state.chart_suggestions_for = None
        st.rerun()


def _action_impact(action: dict, df: pd.DataFrame) -> str | None:
    """Concrete effect of one action, shown before apply ('−47 rows', '−1 column')."""
    action_type = action.get("type")
    col = action.get("column")

    if action_type == "drop_column":
        return "−1 column"
    if action_type == "drop_duplicate_rows":
        n = int(df.duplicated().sum())
        return f"−{n:,} rows" if n else None
    if action_type == "drop_rows_all_null":
        cols = action.get("columns")
        n = int(df.isna().all(axis=1).sum()) if not cols else int(
            df[cols].isna().all(axis=1).sum()
        )
        return f"−{n:,} rows" if n else None
    if not col or col not in df.columns:
        return None
    if action_type == "fill_null":
        n = int(df[col].isna().sum())
        return f"fills {n:,} cells" if n else None
    if action_type == "drop_outlier_rows":
        mask = outlier_mask(
            df[col],
            method=action.get("method", "iqr"),
            factor=float(action.get("factor", 1.5)),
            threshold=float(action.get("threshold", 3.0)),
        )
        n = int(mask.fillna(False).sum())
        return f"−{n:,} rows" if n else "no rows affected"
    if action_type == "cap_outliers":
        lo = float(action.get("lower_percentile", 1)) / 100
        hi = float(action.get("upper_percentile", 99)) / 100
        lower = df[col].quantile(lo)
        upper = df[col].quantile(hi)
        n = int(((df[col] < lower) | (df[col] > upper)).sum())
        return f"caps {n:,} values" if n else "no values affected"
    return None


def _source_badge(action: dict) -> str:
    source = action.get("source")
    if source == "ai":
        return "AI"
    if source == "heuristic":
        return "heuristic"
    return ""


def _render_action_row(action: dict, df: pd.DataFrame) -> None:
    if action.get("type") == "skip":
        st.caption(f"✓ {_action_label(action)}")
        return

    aid = action.get("id", action.get("type"))
    key = f"clean_{aid}"
    if key not in st.session_state:
        st.session_state[key] = True

    err = validate_action(df, action)
    supported = action.get("type") in SUPPORTED_TYPES and err is None
    choices = _resolution_choices(action, df) if supported else []
    col = action.get("column", "")
    reason = action.get("reason", "")

    if not supported:
        label = _action_label(action) + " (not supported)"
        st.checkbox(label, value=False, disabled=True, key=f"disabled_{key}")
        return

    badge = _source_badge(action)
    resolved = _resolved_action(action, df) if len(choices) > 1 else action
    impact = _action_impact(resolved, df)
    meta_bits = [b for b in (badge, impact) if b]
    meta = f" · {' · '.join(meta_bits)}" if meta_bits else ""

    if len(choices) > 1:
        title = f"Column `{col}`" if col else _action_label(action).split(" — ")[0]
        st.checkbox(title, key=key)
        if reason or meta:
            st.caption(f"{reason}{meta}")
        labels = [c[0] for c in choices]
        resolve_key = f"clean_resolve_{aid}"
        default_idx = 0
        if action.get("type") == "fill_null" and action.get("strategy"):
            pref = f"Fill nulls ({action['strategy']})"
            if pref in labels:
                default_idx = labels.index(pref)
        st.selectbox(
            "How to fix",
            labels,
            index=default_idx,
            key=resolve_key,
            label_visibility="visible",
        )
        if resolved.get("type") == "drop_outlier_rows" and col in df.columns:
            mask = outlier_mask(
                df[col],
                method=resolved.get("method", "iqr"),
                factor=float(resolved.get("factor", 1.5)),
                threshold=float(resolved.get("threshold", 3.0)),
            ).fillna(False)
            if mask.any():
                with st.expander(f"Rows that would be removed ({int(mask.sum()):,})"):
                    st.dataframe(df.loc[mask].head(5), use_container_width=True)
    else:
        st.checkbox(f"{_action_label(action)}{meta}", key=key)


def _render_cleaning_workflow(df: pd.DataFrame) -> None:
    if st.session_state.cleaning_applied and st.session_state.df_original is not None:
        st.info("Dataset has been cleaned. Download below or restore the original upload.")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            render_download_csv_button(type_primary=True)
        with c2:
            if st.button("Restore original", use_container_width=True, key="restore_original"):
                _restore_original()
        with c3:
            if st.button("Open Chat tab →", use_container_width=True, key="goto_chat"):
                st.session_state.active_tab = "Chat"
                st.rerun()
        st.markdown("---")

    col_ai, col_fill = st.columns(2)
    with col_ai:
        if st.button("Analyze with AI", use_container_width=True, key="analyze_cleaning"):
            n_cols = len(df.columns)
            with st.spinner(f"Analyzing all {n_cols} columns (batched AI)…"):
                _clear_cleaning_checkbox_keys()
                st.session_state.cleaning_proposal = analyze_for_cleaning(
                    df,
                    st.session_state.schema,
                )
    with col_fill:
        if st.button(
            "Fill all missing → 0%",
            use_container_width=True,
            type="primary",
            key="fill_all_nulls_quick",
        ):
            try:
                _apply_cleaning([{"type": "fill_all_nulls", "id": "fill_all_nulls"}])
            except ValueError as e:
                st.error(str(e))

    proposal = st.session_state.cleaning_proposal
    if proposal is None:
        already_has_download = (
            st.session_state.cleaning_applied and st.session_state.df_original is not None
        )
        if not already_has_download:
            render_download_csv_button()
        return

    st.markdown(proposal.get("summary", ""))

    failed_batches = proposal.get("ai_failed_batches", 0)
    if failed_batches:
        total_batches = proposal.get("ai_total_batches", 0)
        st.warning(
            f"{failed_batches} of {total_batches} AI batches failed (even after retry). "
            "Affected columns use rule-based heuristics instead — look for the "
            "'heuristic' badge below."
        )

    actions = proposal.get("actions", [])

    if not actions:
        st.info("No cleaning actions suggested.")
        if st.button("Dismiss", key="dismiss_empty_proposal"):
            _clear_cleaning_checkbox_keys()
            st.session_state.cleaning_proposal = None
            st.rerun()
        return

    dataset_actions = [a for a in actions if not a.get("column")]
    column_actions = [a for a in actions if a.get("column")]

    if dataset_actions:
        st.markdown('<p class="section-label">Dataset</p>', unsafe_allow_html=True)
        for action in dataset_actions:
            _render_action_row(action, df)

    if column_actions:
        st.markdown(
            f'<p class="section-label">Columns ({len(column_actions)})</p>',
            unsafe_allow_html=True,
        )
        with st.expander("Show all column recommendations", expanded=len(column_actions) <= 12):
            for action in column_actions:
                _render_action_row(action, df)

    selected = _selected_actions(proposal, df)
    if selected:
        validated = []
        for act in selected:
            err = validate_action(df, act)
            if err:
                st.warning(f"Skipped invalid action: {err}")
            else:
                validated.append(act)
        selected = validated

    if selected:
        try:
            preview_df = apply_cleaning_actions(df, selected)
            before = preview_stats(df)
            after = preview_stats(preview_df)
            st.caption(
                f"Preview: {before['rows']:,} → {after['rows']:,} rows · "
                f"{before['columns']} → {after['columns']} cols · "
                f"missing {before['missing_pct']}% → {after['missing_pct']}%"
            )
        except ValueError as e:
            st.warning(f"Preview unavailable: {e}")

    col_apply, col_discard = st.columns(2)
    with col_apply:
        if st.button("Apply selected", use_container_width=True, type="primary", key="apply_cleaning"):
            if not selected:
                st.error("Select at least one action.")
            else:
                try:
                    _apply_cleaning(selected)
                except ValueError as e:
                    st.error(str(e))
    with col_discard:
        if st.button("Discard proposal", use_container_width=True, key="discard_proposal"):
            _clear_cleaning_checkbox_keys()
            st.session_state.cleaning_proposal = None
            st.rerun()


def render_cleaning_page() -> None:
    st.markdown(
        '<p class="chat-header">Clean your data</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="chat-meta">AI suggests fixes — you choose what to apply, then download or explore in Chat.</p>',
        unsafe_allow_html=True,
    )

    df = st.session_state.df
    if df is None:
        st.markdown(
            """
            <div class="landing-hero">
                <p class="hero-kicker">Clean data</p>
                <h1>Prepare <em>your CSV</em></h1>
                <p>Upload a file in the sidebar, then analyze quality issues, apply fixes,
                and download a cleaned export.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"Working on {st.session_state.filename}")
    render_stat_cards(df)
    st.markdown("---")
    _render_cleaning_workflow(df)
