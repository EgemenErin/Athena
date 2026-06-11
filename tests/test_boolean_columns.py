import pandas as pd

from athena.llm.chart_suggestions import _fallback_chart_suggestions, generate_chart_suggestions
from athena.llm.schema import build_schema_string, is_quantitative_dtype, numeric_columns


def test_boolean_excluded_from_numeric_columns():
    df = pd.DataFrame({
        "flag": pd.array([True, False, None, True], dtype="boolean"),
        "amount": [1, 2, 3, 4],
    })
    assert not is_quantitative_dtype(df["flag"].dtype)
    assert "flag" not in numeric_columns(df)
    assert "amount" in numeric_columns(df)


def test_chart_suggestions_with_boolean_column():
    df = pd.DataFrame({
        "flag": pd.array([True, False, None] * 10, dtype="boolean"),
        "score": list(range(30)),
        "category": ["A", "B", "C"] * 10,
    })
    schema = build_schema_string(df)
    assert "boolean" in schema

    specs = _fallback_chart_suggestions(df, n=4)
    assert isinstance(specs, list)

    # Must not raise on nullable boolean columns (Ollama may be absent).
    try:
        generate_chart_suggestions(df, schema, n=3)
    except Exception as exc:
        assert "boolean subtract" not in str(exc)
