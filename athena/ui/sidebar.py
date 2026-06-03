import streamlit as st

from athena.ui.helpers import check_ollama, load_csv_upload, load_suggestions, render_stat_cards


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<p class="brand-title">Athena</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="brand-sub">Ask your CSV anything — runs locally via Ollama.</p>',
            unsafe_allow_html=True,
        )

        if check_ollama():
            st.markdown(
                '<span class="status-pill">Local · qwen2.5-coder</span>',
                unsafe_allow_html=True,
            )
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
                load_csv_upload(uploaded)
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

            st.markdown(
                '<p class="section-label">Suggested questions</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="suggest-hint">Tailored to your columns — click to ask, or refresh for new ideas.</p>',
                unsafe_allow_html=True,
            )

            _col_a, col_b = st.columns([1, 1])
            with col_b:
                refresh = st.button(
                    "↻ New ideas",
                    use_container_width=True,
                    key="refresh_suggestions",
                )

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
