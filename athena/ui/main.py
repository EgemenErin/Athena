import pandas as pd
import streamlit as st

from athena.llm import generate_and_run, summarise_result
from athena.ui.charts import try_make_chart
from athena.ui.helpers import add_to_llm_history


def render_landing() -> None:
    st.markdown(
        """
        <div class="landing-hero">
            <h1>Talk to your data</h1>
            <p>Upload any CSV and ask questions in plain English.
            Athena writes the pandas code, runs it locally, and explains what it found.</p>
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


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return

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


def _handle_question(question: str) -> None:
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


def render_chat() -> None:
    df = st.session_state.df
    st.markdown(
        f'<p class="chat-header">{st.session_state.filename}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="chat-meta">{df.shape[0]:,} rows · {df.shape[1]} columns</p>',
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        _render_message(msg)

    prefill = st.session_state.pop("_prefill", None)
    question = st.chat_input("Ask anything about your data…") or prefill

    if question:
        _handle_question(question)


def render_main() -> None:
    if st.session_state.df is None:
        render_landing()
    else:
        render_chat()
