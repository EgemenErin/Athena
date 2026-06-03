import re
import traceback

import pandas as pd

from athena.llm.schema import categorical_columns, numeric_columns


def extract_code(response_text: str) -> str | None:
    """Pull the first ```python ... ``` block out of the model response."""
    match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = response_text.strip()
    if stripped.startswith("result"):
        return stripped
    return None


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

    return error
