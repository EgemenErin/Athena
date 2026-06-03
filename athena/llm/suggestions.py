import re

import ollama
import pandas as pd

from athena.config import MODEL
from athena.llm.schema import categorical_columns, numeric_columns


def _fallback_suggestions(df: pd.DataFrame, n: int = 5) -> list[str]:
    numeric = numeric_columns(df)
    categorical = categorical_columns(df)
    out: list[str] = []

    if categorical:
        out.append(f"What are the top 10 most common values in '{categorical[0]}'?")
    if len(categorical) > 1:
        out.append(f"How many unique values does '{categorical[1]}' have?")
    if numeric:
        out.append(f"What is the average, min, and max of '{numeric[0]}'?")
    if numeric and categorical:
        out.append(
            f"Show average '{numeric[0]}' grouped by '{categorical[0]}' (top 10)."
        )
    if numeric:
        out.append(f"Which rows have the highest '{numeric[0]}'?")

    generic = [
        "How many rows are in the dataset?",
        "Which columns have the most missing values?",
    ]
    for q in generic:
        if len(out) >= n:
            break
        if q not in out:
            out.append(q)
    return out[:n]


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
    return questions[:n]


def generate_suggested_questions(
    df: pd.DataFrame,
    schema: str,
    n: int = 5,
) -> list[str]:
    """Dataset-specific question suggestions via Ollama."""
    numeric = numeric_columns(df)[:12]
    categorical = categorical_columns(df)[:12]
    schema_excerpt = schema if len(schema) <= 5000 else schema[:5000] + "\n... (truncated)"

    prompt = (
        f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.\n"
        f"Numeric columns: {', '.join(numeric) or 'none'}\n"
        f"Category columns: {', '.join(categorical) or 'none'}\n\n"
        f"Schema:\n{schema_excerpt}\n\n"
        f"Write exactly {n} short plain-English questions answerable with pandas on this data. "
        "Use real column names from the schema. Mix top-N, averages, counts, and comparisons.\n"
        "Numbered list only, no preamble:\n"
        "1. ...\n2. ..."
    )

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.65, "num_predict": 400},
        )
        parsed = _parse_question_list(response["message"]["content"].strip(), n)
        if len(parsed) >= max(3, n // 2):
            return parsed[:n]
    except Exception:
        pass

    return _fallback_suggestions(df, n)
