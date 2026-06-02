
"""
llm_engine.py
-------------
Core LLM logic for the data chatbot — Ollama local backend.
Handles schema building, code generation, sandboxed execution,
and narrative summarisation.

Requirements:
    pip install ollama pandas
    ollama pull qwen2.5-coder:7b
"""

import re
import traceback
import ollama
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL = "qwen2.5-coder:7b"   # fits in 8GB VRAM; swap to :14b if you want more power
MAX_RETRIES = 1               # one automatic fix attempt on code error

# ── Schema builder ──────────────────────────────────────────────────────────────

def build_schema_string(df: pd.DataFrame, max_values: int = 10) -> str:
    """
    Produces a compact schema description of the DataFrame to inject
    into the system prompt so the model knows exactly what columns exist.

    Numeric columns  → show min/max range + null count
    Categorical cols → show sample unique values + null count
    """
    lines = [f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"]

    for col in df.columns:
        dtype = df[col].dtype
        null_count = int(df[col].isna().sum())

        if pd.api.types.is_numeric_dtype(dtype):
            col_min = df[col].min()
            col_max = df[col].max()
            lines.append(
                f"Column: '{col}'  dtype: {dtype}\n"
                f"  range: {col_min} – {col_max}  |  nulls: {null_count}"
            )
        else:
            samples = df[col].dropna().unique()[:max_values]
            sample_str = ", ".join(str(v) for v in samples)
            n_unique = df[col].nunique()
            if n_unique > max_values:
                sample_str += f", ... ({n_unique} unique)"
            lines.append(
                f"Column: '{col}'  dtype: {dtype}\n"
                f"  sample values: {sample_str}  |  nulls: {null_count}"
            )

    return "\n".join(lines)


# ── System prompt ───────────────────────────────────────────────────────────────

def build_system_prompt(schema: str) -> str:
    return f"""You are a data analyst assistant. You have access to a pandas DataFrame called `df`.

Here is the schema of `df`:
{schema}

When the user asks a question about the data, respond with ONLY a Python code block.

Rules:
- Use ONLY `df` and `pd` — both are already available, do not import anything
- Assign the final answer to a variable called `result`
- `result` must be a DataFrame, a Series, or a scalar (int / float / str)
- Do NOT use print(), display(), or any output functions
- Do NOT import any libraries
- When filtering rows, always use .copy() to avoid SettingWithCopyWarning
- For "top N" questions return a sorted DataFrame or Series
- For single-number answers (count, mean, max) assign the scalar directly

Respond with ONLY the code block and nothing else — no explanation, no preamble.

Example:
```python
result = (
    df[df["ConvertedCompYearly"] > 0]
    .groupby("Country")["ConvertedCompYearly"]
    .median()
    .sort_values(ascending=False)
    .head(10)
)
```
"""


# ── Code extraction ─────────────────────────────────────────────────────────────

def extract_code(response_text: str) -> str | None:
    """Pull the first ```python ... ``` block out of the model response."""
    # try labelled fence first
    match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: unlabelled fence
    match = re.search(r"```\s*(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # last resort: the whole response might just be raw code
    stripped = response_text.strip()
    if stripped.startswith("result"):
        return stripped
    return None


# ── Sandboxed execution ─────────────────────────────────────────────────────────

def run_code(code: str, df: pd.DataFrame) -> tuple:
    """
    Execute generated code in an isolated scope.
    Only `df` and `pd` are available — nothing else.
    Returns (result, error_string).
    """
    sandbox = {"df": df.copy(), "pd": pd}
    try:
        exec(code, sandbox)  # noqa: S102
        result = sandbox.get("result", None)
        if result is None:
            return None, "Code ran but did not assign anything to `result`."
        return result, None
    except Exception:
        return None, traceback.format_exc()


# ── Core generation + retry loop ────────────────────────────────────────────────

def generate_and_run(
    question: str,
    df: pd.DataFrame,
    schema: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Main entry point called by the Streamlit app.

    Flow:
      1. Send user question to Ollama with schema in the system prompt
      2. Extract the returned Python code block
      3. Execute it in a sandbox against the real DataFrame
      4. On failure, send the error back for one automatic fix attempt
      5. Return a result dict

    chat_history: list of {"role": "user"|"assistant", "content": "..."}
                  pass st.session_state.messages for multi-turn support
                  so follow-ups like "now filter that to Europe" work correctly
    """
    system_prompt = build_system_prompt(schema)
    history = chat_history or []

    messages = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )

    # ── First attempt ────────────────────────────────────────────────────────
    response = ollama.chat(model=MODEL, messages=messages)
    raw = response["message"]["content"]
    code = extract_code(raw)

    if not code:
        return {
            "code": None,
            "result": None,
            "error": "Model did not return a recognisable code block.",
            "raw_response": raw,
        }

    result, error = run_code(code, df)

    # ── One retry if execution failed ────────────────────────────────────────
    if error and MAX_RETRIES > 0:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    f"That code raised an error:\n\n{error}\n\n"
                    "Fix it and return only the corrected ```python block."
                ),
            },
        ]
        retry_response = ollama.chat(model=MODEL, messages=retry_messages)
        raw = retry_response["message"]["content"]
        code = extract_code(raw)
        if code:
            result, error = run_code(code, df)

    return {
        "code": code,
        "result": result,
        "error": error,
        "raw_response": raw,
    }


# ── Narrative summariser ─────────────────────────────────────────────────────────

def summarise_result(question: str, result) -> str:
    """
    Second LLM call: takes the question + query result and returns
    a 1–2 sentence plain-English insight with specific numbers.
    This is what makes the tool feel like a data analyst, not a query engine.
    """
    if isinstance(result, pd.DataFrame):
        result_preview = result.head(10).to_string(index=False)
    elif isinstance(result, pd.Series):
        result_preview = result.head(10).to_string()
    else:
        result_preview = str(result)

    prompt = (
        f'A user asked: "{question}"\n\n'
        f"The data query returned:\n{result_preview}\n\n"
        "Write 1–2 sentences summarising the key insight. "
        "Be specific — mention actual numbers or names from the data. "
        "Do not say 'the data shows' or 'based on the results'."
    )

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


# ── Quick smoke test (run this file directly to verify Ollama is working) ────────

if __name__ == "__main__":
    print("Running smoke test against Ollama...\n")

    sample_df = pd.DataFrame({
        "Country": ["USA", "Germany", "Poland", "USA", "Germany"],
        "LanguageHaveWorkedWith": ["Python;SQL", "Python;Java", "Python", "JavaScript", "Python;SQL"],
        "ConvertedCompYearly": [120000, 85000, 45000, 95000, 90000],
        "YearsCodePro": [5, 8, 2, 3, 10],
    })

    schema = build_schema_string(sample_df)
    print("Schema:\n", schema, "\n")

    question = "What is the average yearly compensation by country?"
    print(f"Question: {question}\n")

    output = generate_and_run(question, sample_df, schema)

    if output["error"]:
        print("ERROR:", output["error"])
    else:
        print("Generated code:\n", output["code"])
        print("\nResult:\n", output["result"])
        print("\nNarrative:", summarise_result(question, output["result"]))