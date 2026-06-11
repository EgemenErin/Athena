import builtins
import json
import re
import threading
import traceback

import pandas as pd

from athena.llm.schema import categorical_columns, coerce_numeric_like_columns, numeric_columns

EXECUTION_TIMEOUT_SECONDS = 10

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "int", "isinstance", "issubclass", "len",
    "list", "map", "max", "min", "next", "range", "repr", "reversed", "round",
    "set", "sorted", "str", "sum", "tuple", "zip", "ValueError", "TypeError",
    "KeyError", "IndexError", "ZeroDivisionError", "Exception", "True",
    "False", "None",
)

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(builtins, name)
}

_BLOCKED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"^\s*(import|from)\s", re.MULTILINE),
        "imports are not allowed — only `df` and `pd` are available",
    ),
    (
        re.compile(r"__"),
        "double-underscore attributes are not allowed",
    ),
    (
        re.compile(r"\bopen\s*\("),
        "file access is not allowed",
    ),
    (
        re.compile(r"\b(os|sys|subprocess|shutil|socket|pathlib)\s*\."),
        "system modules are not allowed",
    ),
    (
        re.compile(r"\b(eval|exec|compile|globals|locals|input|breakpoint)\s*\("),
        "dynamic execution is not allowed",
    ),
]


def validate_code_safety(code: str) -> str | None:
    """Return an error message when generated code uses blocked constructs."""
    for pattern, reason in _BLOCKED_PATTERNS:
        if pattern.search(code):
            return f"Generated code was blocked for safety: {reason}."
    return None


def extract_code(response_text: str) -> str | None:
    """
    Pull generated code out of the model response.
    Prefers structured JSON {"code": "..."}; falls back to fenced blocks.
    """
    text = response_text.strip()

    json_candidates = [text]
    fence_json = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_json:
        json_candidates.insert(0, fence_json.group(1))
    for candidate in json_candidates:
        if not candidate.startswith("{"):
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("code"), str) and data["code"].strip():
            return data["code"].strip()

    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.startswith("result"):
        return text
    return None


def run_code(
    code: str,
    df: pd.DataFrame,
    timeout: float = EXECUTION_TIMEOUT_SECONDS,
) -> tuple:
    """
    Execute generated code in an isolated scope with a hard timeout.
    Only `df` and `pd` are available — nothing else.
    Returns (result, error_string).
    """
    safety_error = validate_code_safety(code)
    if safety_error:
        return None, safety_error

    sandbox = {
        "df": coerce_numeric_like_columns(df),
        "pd": pd,
        "__builtins__": SAFE_BUILTINS,
    }
    outcome: dict = {}

    def _target() -> None:
        try:
            exec(code, sandbox)  # noqa: S102
            outcome["result"] = sandbox.get("result", None)
        except Exception:
            outcome["error"] = traceback.format_exc()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, (
            f"The analysis took longer than {timeout:g} seconds and was stopped. "
            "Try a simpler question, or filter the data down first."
        )
    if "error" in outcome:
        return None, outcome["error"]
    result = outcome.get("result")
    if result is None:
        return None, "Code ran but did not assign anything to `result`."
    return result, None


def friendly_execution_error(error: str, df: pd.DataFrame) -> str:
    """Convert noisy pandas tracebacks into clear user-facing hints."""
    missing_col = re.search(r"KeyError: ['\"]([^'\"]+)['\"]", error)
    if missing_col:
        requested = missing_col.group(1)
        sample = ", ".join(df.columns[:12])
        if len(df.columns) > 12:
            sample += ", ..."
        return (
            f"Column '{requested}' was not found in the dataset. "
            f"Use exact column names from the schema. "
            f"Example available columns: {sample}"
        )

    numeric_cols = numeric_columns(df)
    categorical_cols = categorical_columns(df)
    numeric_hint = ", ".join(numeric_cols[:8]) if numeric_cols else "no numeric columns detected"
    group_hint = ", ".join(categorical_cols[:8]) if categorical_cols else "no categorical columns detected"

    if "does not support operation 'mean'" in error and "dtype 'str'" in error:
        return (
            "Tried to calculate an average on text values. "
            f"Use a numeric column for mean/avg. Numeric examples: {numeric_hint}. "
            f"Group-by examples: {group_hint}."
        )

    unsupported_reduce = re.search(
        r"Cannot perform reduction '([^']+)' with string dtype",
        error,
    )
    if unsupported_reduce:
        op = unsupported_reduce.group(1)
        return (
            f"Tried to run `{op}` on text values. "
            f"Use a numeric column for `{op}`. Numeric examples: {numeric_hint}. "
            f"If grouping, use text columns only as group keys (e.g. {group_hint})."
        )

    if "Invalid comparison between dtype=str" in error or "Invalid comparison between dtype=string" in error:
        return (
            "Tried to compare a text column to a number. "
            f"Use pd.to_numeric(df['column'], errors='coerce') before filtering, or pick a numeric column. "
            f"Numeric examples: {numeric_hint}."
        )

    if "ArrowNotImplementedError" in error and "greater" in error:
        return (
            "Tried to compare a text column to a number. "
            f"Coerce with pd.to_numeric(..., errors='coerce') first, or use: {numeric_hint}."
        )

    if "has no attribute 'append'" in error:
        return (
            "Generated code used DataFrame.append(), which is not supported in this pandas version. "
            "Use pd.concat([df1, df2], ignore_index=True) to combine tables instead."
        )

    return error
