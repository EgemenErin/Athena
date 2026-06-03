import pandas as pd

from athena.ui.charts import build_chart_from_spec, should_chart, try_make_chart


def test_correlation_matrix_no_chart():
    corr = pd.DataFrame(
        [[1.0, 0.07], [0.07, 1.0]],
        columns=["ConvertedCompYearly", "JobSat"],
        index=["ConvertedCompYearly", "JobSat"],
    )
    assert should_chart(corr) is False
    assert try_make_chart(corr) is None


def test_groupby_bar_chart():
    df = pd.DataFrame({
        "Country": ["USA", "Germany", "Poland"],
        "median_pay": [120000, 85000, 45000],
    })
    assert should_chart(df) is True
    assert try_make_chart(df) is not None


def test_scalar_no_chart():
    assert should_chart(42) is False
    assert try_make_chart(42) is None


def test_series_chart():
    s = pd.Series([10, 20, 30], index=["a", "b", "c"])
    assert should_chart(s) is True


def test_build_chart_from_spec_uses_theme():
    df = pd.DataFrame({"Age": [25, 30, 35, 40, 45]})
    spec = {
        "chart_type": "histogram",
        "x": "Age",
        "title": "Ages",
        "rationale": "Distribution",
    }
    fig = build_chart_from_spec(df, spec)
    assert fig is not None
    assert fig.layout.paper_bgcolor == "#14161c"
