import re

import ollama
import pandas as pd

from athena.config import MODEL
from athena.llm.personas import BUSINESS_ANALYST_RULES
from athena.llm.schema import (
    build_column_index,
    categorical_columns,
    comparable_numeric_columns,
    numeric_columns,
)
from athena.llm.suggestion_validate import (
    columns_for_topic,
    filter_answerable_suggestions,
    is_suggestion_answerable,
)


def _meaningful_categorical_columns(df: pd.DataFrame, max_cols: int = 8) -> list[str]:
    """Prefer categoricals with useful cardinality for questions."""
    scored: list[tuple[int, str]] = []
    n = len(df)
    for col in categorical_columns(df):
        nunique = df[col].nunique(dropna=True)
        if nunique < 2 or (n > 0 and nunique > n * 0.95):
            continue
        scored.append((nunique, col))
    scored.sort(key=lambda x: (-min(x[0], 50), x[1]))
    return [c for _, c in scored[:max_cols]]


def _experience_numeric_columns(df: pd.DataFrame) -> list[str]:
    priority = []
    for col in comparable_numeric_columns(df):
        cl = col.lower()
        if any(k in cl for k in ("yearscode", "workexp", "yearsexp")):
            priority.append(col)
    for col in comparable_numeric_columns(df):
        cl = col.lower()
        if col in priority:
            continue
        if "year" in cl or "exp" in cl:
            priority.append(col)
    return priority


def _fallback_suggestions(df: pd.DataFrame, n: int = 5) -> list[str]:
    numeric = numeric_columns(df)
    categorical = _meaningful_categorical_columns(df)
    out: list[str] = []

    exp_cols = _experience_numeric_columns(df)
    if exp_cols:
        out.append(f"How many rows have `{exp_cols[0]}` greater than 10?")

    if categorical:
        col = categorical[0]
        out.append(f"What are the top 10 most common values in `{col}`?")
    if len(categorical) > 1:
        out.append(f"How many unique values does `{categorical[1]}` have?")
    if numeric and categorical:
        out.append(
            f"What is the median `{numeric[0]}` grouped by `{categorical[0]}` (top 10)?"
        )
    elif numeric:
        out.append(f"What is the median, min, and max of `{numeric[0]}`?")

    generic = [
        "How many rows are in the dataset?",
        "Which columns have the highest percentage of missing values?",
    ]
    for q in generic:
        if len(out) >= n:
            break
        if q not in out:
            out.append(q)

    return filter_answerable_suggestions(out, df, n)


def _parse_question_list(raw: str, n: int) -> list[str]:
    questions: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[-*]\s*", "", line)
        cleaned = re.sub(r"^\d+[\.\):]\s*", "", cleaned)
        cleaned = cleaned.strip(chr(34) + chr(39))
        if len(cleaned) > 12:
            questions.append(cleaned if cleaned.endswith("?") else cleaned + "?")
    return questions[: n * 2]


def generate_suggested_questions(
    df: pd.DataFrame,
    schema: str,
    n: int = 5,
) -> list[str]:
    """Dataset-specific question suggestions via Ollama, validated against columns."""
    numeric = numeric_columns(df)[:20]
    categorical = _meaningful_categorical_columns(df, max_cols=12)
    column_index = build_column_index(df)
    schema_excerpt = schema if len(schema) <= 4000 else schema[:4000] + "\n... (truncated)"

    prompt = (
        f"{BUSINESS_ANALYST_RULES}\n\n"
        f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.\n"
        f"Numeric columns (sample): {', '.join(f'`{c}`' for c in numeric) or 'none'}\n"
        f"Category columns (sample): {', '.join(f'`{c}`' for c in categorical) or 'none'}\n\n"
        f"{column_index}\n\n"
        f"Schema detail:\n{schema_excerpt}\n\n"
        f"Write exactly {n} questions. Each MUST cite at least one column from the list above using backticks.\n"
        "Numbered list only, no preamble:\n"
        "1. ...\n2. ..."
    )

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.35, "num_predict": 600},
        )
        parsed = _parse_question_list(response["message"]["content"].strip(), n)
        validated = filter_answerable_suggestions(parsed, df, n)
        if len(validated) >= max(3, n // 2):
            return validated[:n]
    except Exception:
        pass

    return _fallback_suggestions(df, n)
