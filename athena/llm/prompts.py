import pandas as pd

from athena.llm.personas import SENIOR_ANALYST_RULES
from athena.llm.schema import categorical_columns, comparable_numeric_columns


def _pick_example_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Top categorical, numeric, and multi-value columns from the real dataset."""
    from athena.llm.chart_transforms import multivalue_columns

    numeric = comparable_numeric_columns(df)
    numeric_col = numeric[0] if numeric else None

    multi = multivalue_columns(df, max_cols=1)
    multi_col = multi[0] if multi else None

    cat_col = None
    best_score = -1.0
    n = len(df)
    for col in categorical_columns(df):
        if col == multi_col:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique < 2 or (n > 0 and nunique > n * 0.9):
            continue
        # Prefer low-to-mid cardinality grouping columns.
        score = 2.0 if 3 <= nunique <= 25 else 1.0
        if score > best_score:
            best_score = score
            cat_col = col

    return cat_col, numeric_col, multi_col


def build_fewshot_examples(df: pd.DataFrame | None) -> str:
    """2–3 worked examples using the uploaded dataset's actual columns."""
    cat_col = num_col = multi_col = None
    if df is not None and len(df.columns):
        cat_col, num_col, multi_col = _pick_example_columns(df)

    # Generic placeholders when the dataset lacks a suitable column type.
    cat = cat_col or "category_column"
    num = num_col or "numeric_column"

    examples = [
        f"""Example (grouped aggregation):
{{"code": "result = (\\n    df.groupby(\\"{cat}\\")[\\"{num}\\"]\\n    .median()\\n    .sort_values(ascending=False)\\n    .head(10)\\n    .reset_index()\\n)"}}""",
        f"""Example (count rows after a filter):
{{"code": "vals = pd.to_numeric(df[\\"{num}\\"], errors=\\"coerce\\")\\nresult = int((vals > vals.median()).sum())"}}""",
    ]

    if multi_col:
        examples.append(
            f"""Example (count distinct values in a multi-select column):
{{"code": "items = df[\\"{multi_col}\\"].dropna().str.split(\\";\\").explode().str.strip()\\nresult = items.nunique()"}}"""
        )
    else:
        examples.append(
            f"""Example (top values in a category):
{{"code": "result = df[\\"{cat}\\"].value_counts().head(10).reset_index()"}}"""
        )

    return "\n\n".join(examples)


def build_system_prompt(schema: str, df: pd.DataFrame | None = None) -> str:
    return f"""{SENIOR_ANALYST_RULES}

Here is the schema of `df`:
{schema}

{build_fewshot_examples(df)}
"""
