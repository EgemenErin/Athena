import pandas as pd

from athena.llm.schema import (
    coerce_numeric_like_columns,
    comparable_numeric_columns,
    looks_numeric_string,
)


def test_string_years_column_is_comparable():
    df = pd.DataFrame({"YearsCodePro": ["5", "12", "20", None]})
    assert looks_numeric_string(df["YearsCodePro"])
    assert "YearsCodePro" in comparable_numeric_columns(df)


def test_pure_text_not_comparable():
    df = pd.DataFrame({"Country": ["USA", "DE", "FR"]})
    assert "Country" not in comparable_numeric_columns(df)


def test_coerce_enables_numeric_compare():
    df = pd.DataFrame({"YearsCode": ["5", "12", "20"]})
    coerced = coerce_numeric_like_columns(df)
    assert pd.api.types.is_numeric_dtype(coerced["YearsCode"])
    assert (coerced["YearsCode"] > 10).sum() == 2
