import pandas as pd

from athena.llm.chart_transforms import (
    bin_column,
    coerce_column,
    explode_column,
    is_multivalue_column,
)
from athena.ui.charts import build_chart_from_spec, prepare_chart_data


def test_coerce_numeric_string_histogram():
    df = pd.DataFrame({"Salary": ["50000", "60000", "70000", "80000", "90000"]})
    spec = {
        "chart_type": "histogram",
        "x": "Salary",
        "transform_x": "coerce",
        "title": "Salary spread",
        "rationale": "Parsed from text",
    }
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    assert len(plot_df) == 5
    assert build_chart_from_spec(df, spec) is not None


def test_explode_multivalue_bar():
    df = pd.DataFrame({
        "Skills": ["Python;SQL", "Java;Python", "SQL", "Python;Java;SQL", "R"],
    })
    assert is_multivalue_column(df, "Skills") is True
    spec = {
        "chart_type": "bar",
        "x": "Skills",
        "y": "count",
        "aggregation": "count",
        "transform_x": "explode",
        "title": "Top skills",
        "rationale": "Exploded list",
    }
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    assert plot_df["count"].sum() >= 5
    assert "Python" in plot_df.iloc[:, 0].astype(str).values


def test_explode_handles_missing_cells():
    df = pd.DataFrame({
        "Skills": ["Python;SQL", None, "Java", float("nan"), "SQL;R"],
    })
    work, _ = explode_column(df, "Skills")
    assert len(work) >= 4
    assert work["Skills"].notna().all()


def test_bin_high_cardinality():
    df = pd.DataFrame({"Score": list(range(100))})
    work, binned_col = bin_column(df, "Score", n_bins=8)
    assert binned_col == "Score (binned)"
    assert work[binned_col].nunique() <= 10
