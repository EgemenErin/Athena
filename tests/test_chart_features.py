"""Phase 2 chart features: Other bucket, boolean flags, aggregations, line axes."""

import pandas as pd

from athena.llm.chart_suggestions import (
    _collect_fallback_candidates,
    _is_continuous_metric,
    _is_ordered_line_axis,
    validate_chart_spec,
)
from athena.llm.schema import boolean_flag_columns, is_boolean_like
from athena.ui.charts import build_chart_from_spec, chart_truncation_note, prepare_chart_data


def _flag_df(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        "Arrest": [1, 0, 1] * (n // 3),
        "Verified": pd.array([True, False, None] * (n // 3), dtype="boolean"),
        "Region": ["North", "South", "East"] * (n // 3),
        "Amount": [float(i * 10) for i in range(n)],
    })


# ---------- boolean flag detection ----------

def test_is_boolean_like_covers_bool_nullable_and_01():
    df = _flag_df()
    assert is_boolean_like(df["Arrest"])
    assert is_boolean_like(df["Verified"])
    assert not is_boolean_like(df["Amount"])
    assert not is_boolean_like(df["Region"])


def test_boolean_flag_columns():
    flags = boolean_flag_columns(_flag_df())
    assert "Arrest" in flags
    assert "Verified" in flags
    assert "Amount" not in flags


def test_flags_are_not_continuous_metrics():
    df = _flag_df()
    assert not _is_continuous_metric(df, "Arrest")
    assert not _is_continuous_metric(df, "Verified")


def test_mean_bar_of_flag_rejected():
    df = _flag_df()
    spec = {
        "chart_type": "bar",
        "x": "Region",
        "y": "Arrest",
        "title": "Average arrest",
        "rationale": "bad — flags are rates",
    }
    assert validate_chart_spec(df, spec) is False


def test_pct_true_bar_accepted_and_renders():
    df = _flag_df()
    spec = {
        "chart_type": "bar",
        "x": "Region",
        "y": "Arrest",
        "aggregation": "pct_true",
        "title": "% arrests by region",
        "rationale": "Flag rate per region",
    }
    assert validate_chart_spec(df, spec) is True
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    assert plot_df["Arrest"].between(0, 100).all()
    assert build_chart_from_spec(df, spec) is not None


def test_fallback_suggests_pct_true_for_flags():
    candidates = _collect_fallback_candidates(_flag_df())
    assert any(c.get("aggregation") == "pct_true" for c in candidates)


# ---------- aggregations ----------

def test_bar_aggregations_sum_median_min_max():
    df = pd.DataFrame({
        "Cat": ["a", "a", "b", "b"],
        "Val": [1.0, 3.0, 10.0, 20.0],
    })
    for agg, expected_a in [("sum", 4.0), ("median", 2.0), ("min", 1.0), ("max", 3.0)]:
        spec = {"chart_type": "bar", "x": "Cat", "y": "Val", "aggregation": agg}
        plot_df = prepare_chart_data(df, spec)
        assert plot_df is not None
        got = plot_df.set_index("Cat")["Val"]["a"]
        assert got == expected_a, f"{agg}: {got} != {expected_a}"


# ---------- Other bucket / truncation ----------

def test_count_bar_groups_remainder_into_other():
    df = pd.DataFrame({"City": [f"City{i}" for i in range(40)] * 2})
    spec = {"chart_type": "bar", "x": "City", "y": "count", "aggregation": "count"}
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    labels = plot_df["City"].astype(str).tolist()
    assert any(label.startswith("Other (") for label in labels)
    assert plot_df["count"].sum() == 80  # nothing silently dropped
    assert "truncation_note" in plot_df.attrs

    fig = build_chart_from_spec(df, spec)
    assert fig is not None
    note = chart_truncation_note(fig)
    assert note and "Showing top" in note


def test_pie_groups_remainder_into_other():
    df = pd.DataFrame({"Kind": [f"K{i}" for i in range(30)] * 3})
    spec = {"chart_type": "pie", "x": "Kind", "title": "Kinds", "rationale": "share"}
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    assert len(plot_df) <= 10
    assert any(str(v).startswith("Other (") for v in plot_df.iloc[:, 0])
    assert plot_df["count"].sum() == 90


def test_small_count_bar_has_no_other():
    df = pd.DataFrame({"Cat": ["a", "b", "c"] * 5})
    spec = {"chart_type": "bar", "x": "Cat", "y": "count", "aggregation": "count"}
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    assert not any(str(v).startswith("Other") for v in plot_df["Cat"])
    assert "truncation_note" not in plot_df.attrs


def test_mean_bar_truncates_with_note():
    df = pd.DataFrame({
        "City": [f"City{i}" for i in range(40)],
        "Pay": [float(50_000 + i * 997) for i in range(40)],
    })
    spec = {"chart_type": "bar", "x": "City", "y": "Pay", "aggregation": "mean"}
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    assert len(plot_df) == 25
    assert not any(str(v).startswith("Other") for v in plot_df["City"])
    assert "Showing top 25 of 40" in plot_df.attrs["truncation_note"]


# ---------- line chart axes ----------

def test_month_names_are_ordered_line_axis():
    df = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"] * 5,
        "Sales": list(range(30)),
    })
    assert _is_ordered_line_axis(df, "Month") is True
    spec = {
        "chart_type": "line",
        "x": "Month",
        "y": "count",
        "aggregation": "count",
        "title": "Rows by month",
        "rationale": "trend",
    }
    assert validate_chart_spec(df, spec) is True
    plot_df = prepare_chart_data(df, spec)
    assert plot_df is not None
    assert plot_df["Month"].tolist() == ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]


def test_numeric_sequence_is_ordered_line_axis():
    df = pd.DataFrame({
        "Week": list(range(1, 27)) * 2,
        "Visits": list(range(52)),
    })
    assert _is_ordered_line_axis(df, "Week") is True


def test_free_text_is_not_line_axis():
    df = pd.DataFrame({
        "Name": [f"person{i}" for i in range(30)],
        "Score": list(range(30)),
    })
    assert _is_ordered_line_axis(df, "Name") is False
