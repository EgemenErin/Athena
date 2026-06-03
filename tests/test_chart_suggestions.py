import numpy as np
import pandas as pd

from athena.llm.chart_suggestions import (
    _fallback_chart_suggestions,
    _is_coded_numeric_dimension,
    _is_likely_identifier,
    _normalize_spec,
    chart_description,
    suggest_charts_for_columns,
    validate_chart_spec,
)
from athena.ui.charts import build_chart_from_spec, prepare_chart_data


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Country": ["USA", "Germany", "Poland", "France", "Spain"],
        "Salary": [120000, 85000, 45000, 70000, 55000],
        "Age": [34, 29, 41, 38, 31],
        "Score": [88.5, 72.0, 65.5, 79.0, 81.2],
    })


def test_fallback_produces_valid_specs():
    df = _sample_df()
    specs = _fallback_chart_suggestions(df, n=6)
    assert len(specs) >= 2
    for spec in specs:
        assert validate_chart_spec(df, spec)


def test_validate_rejects_unknown_column():
    df = _sample_df()
    bad = {
        "chart_type": "bar",
        "x": "NotAColumn",
        "y": "Salary",
        "title": "Bad",
        "rationale": "Invalid x",
    }
    assert validate_chart_spec(df, bad) is False


def test_validate_rejects_high_cardinality_pie():
    df = pd.DataFrame({"id": range(20)})
    spec = {
        "chart_type": "pie",
        "x": "id",
        "y": None,
        "title": "Too many",
        "rationale": "Many categories",
    }
    assert validate_chart_spec(df, spec) is False


def test_build_chart_from_spec_bar():
    df = _sample_df()
    spec = {
        "chart_type": "bar",
        "x": "Country",
        "y": "Salary",
        "title": "Pay by country",
        "rationale": "Compare salaries",
    }
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    assert len(plot_df) == 5
    fig = build_chart_from_spec(df, spec)
    assert fig is not None
    assert fig.layout.paper_bgcolor == "#14161c"


def test_response_id_not_suggested():
    df = pd.DataFrame({
        "ResponseId": range(500),
        "YearsCode": [5, 10, 15, 20, 25] * 100,
        "Country": ["USA", "UK", "DE", "FR", "CA"] * 100,
        "ConvertedCompYearly": [50_000 + i * 100 for i in range(500)],
    })
    assert _is_likely_identifier(df, "ResponseId") is True
    specs = _fallback_chart_suggestions(df, n=6)
    hist_x = [s["x"] for s in specs if s["chart_type"] == "histogram"]
    assert "ResponseId" not in hist_x
    assert any(s["chart_type"] == "bar" for s in specs)


def test_coded_dimensions_not_scatter():
    df = pd.DataFrame({
        "YEAR": [2000, 2001, 2002, 2003, 2004] * 20,
        "DISTRICT": list(range(1, 6)) * 20,
        "COMMUNITY_AREA_NUMBER": list(range(1, 11)) * 10,
        "FBICODE": [15, 16, 17, 18] * 25,
    })
    assert _is_coded_numeric_dimension(df, "DISTRICT") is True
    assert _is_coded_numeric_dimension(df, "COMMUNITY_AREA_NUMBER") is True
    bad = {
        "chart_type": "scatter",
        "x": "YEAR",
        "y": "DISTRICT",
        "title": "bad",
        "rationale": "bad",
    }
    assert validate_chart_spec(df, bad) is False
    specs = _fallback_chart_suggestions(df, n=6)
    assert not any(
        s["chart_type"] == "scatter"
        and s.get("x") in ("YEAR", "DISTRICT", "COMMUNITY_AREA_NUMBER", "FBICODE")
        for s in specs
    )


def test_validate_rejects_id_histogram():
    df = pd.DataFrame({"ResponseId": range(100)})
    spec = {
        "chart_type": "histogram",
        "x": "ResponseId",
        "title": "ID hist",
        "rationale": "bad",
    }
    assert validate_chart_spec(df, spec) is False


def test_six_distinct_groupings_and_chart_types():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "YEAR": [2000, 2001, 2002, 2003, 2004] * 20,
        "DISTRICT": list(range(1, 6)) * 20,
        "COMMUNITY_AREA_NUMBER": list(range(1, 11)) * 10,
        "FBICODE": [15, 16, 17, 18] * 25,
        "ConvertedCompYearly": rng.choice(
            [40_000, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000],
            100,
        ),
        "YearsCode": rng.integers(1, 30, 100),
        "WorkExp": rng.integers(1, 40, 100),
    })
    specs = _fallback_chart_suggestions(df, n=6)
    assert len(specs) == 6
    breakdown_x = [s["x"] for s in specs if s.get("chart_type") in ("bar", "pie", "line") and s.get("x")]
    assert len(breakdown_x) == len(set(breakdown_x))
    assert len({s["chart_type"] for s in specs}) >= 4


def test_build_chart_from_spec_histogram():
    df = _sample_df()
    spec = {
        "chart_type": "histogram",
        "x": "Age",
        "y": None,
        "title": "Age distribution",
        "rationale": "Spread of ages",
    }
    fig = build_chart_from_spec(df, spec)
    assert fig is not None


def test_normalize_spec_fills_missing_rationale():
    spec = _normalize_spec({
        "chart_type": "bar",
        "x": "Country",
        "y": "Salary",
        "title": "Pay by country",
    })
    assert spec is not None
    assert spec["rationale"]
    assert chart_description(spec) == spec["rationale"]


def test_suggest_charts_for_country_salary():
    df = _sample_df()
    options = suggest_charts_for_columns(df, "Country", "Salary")
    assert len(options) >= 1
    assert any(s["chart_type"] == "bar" and s["x"] == "Country" for s in options)
    for spec in options:
        assert validate_chart_spec(df, spec)


def test_suggest_charts_for_numeric_x_only():
    df = _sample_df()
    options = suggest_charts_for_columns(df, "Age")
    assert len(options) >= 1
    assert any(s["chart_type"] == "histogram" and s["x"] == "Age" for s in options)
    for spec in options:
        assert validate_chart_spec(df, spec)
