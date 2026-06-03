import pandas as pd
import pytest

from athena.llm.cleaning import (
    _fallback_proposal,
    apply_cleaning_actions,
    preview_stats,
    proposal_fill_all_nulls,
    validate_action,
)
from athena.llm.cleaning_columns import (
    build_per_column_proposal,
    column_profile,
    heuristic_column_action,
)


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "id": [1, 1, 2, 3],
        "name": ["a", "a", None, "c"],
        "score": [10.0, 10.0, None, 30.0],
        "empty_col": [None, None, None, None],
    })


def test_apply_drop_column_and_fill_null(messy_df):
    actions = [
        {"id": "1", "type": "drop_column", "column": "empty_col"},
        {"id": "2", "type": "fill_null", "column": "score", "strategy": "median"},
    ]
    result = apply_cleaning_actions(messy_df, actions)
    assert "empty_col" not in result.columns
    assert result["score"].isna().sum() == 0


def test_apply_drop_duplicates(messy_df):
    actions = [{"id": "1", "type": "drop_duplicate_rows"}]
    result = apply_cleaning_actions(messy_df, actions)
    assert result.shape[0] < messy_df.shape[0]


def test_validate_unknown_column(messy_df):
    err = validate_action(messy_df, {"type": "drop_column", "column": "nope"})
    assert err is not None


def test_fallback_proposal_has_actions(messy_df):
    proposal = _fallback_proposal(messy_df)
    assert "summary" in proposal
    assert isinstance(proposal["actions"], list)
    assert len(proposal["actions"]) > 0


def test_preview_stats(messy_df):
    stats = preview_stats(messy_df)
    assert stats["rows"] == 4
    assert stats["columns"] == 4
    assert stats["missing_pct"] > 0


def test_drop_outlier_rows():
    df = pd.DataFrame({
        "Country": ["Gabon", "USA", "UK", "DE"],
        "ConvertedCompYearly": [2_000_000, 143_000, 95_000, 90_000],
    })
    actions = [{
        "id": "1",
        "type": "drop_outlier_rows",
        "column": "ConvertedCompYearly",
        "method": "iqr",
        "factor": 1.5,
    }]
    result = apply_cleaning_actions(df, actions)
    assert "Gabon" not in result["Country"].values
    assert result["ConvertedCompYearly"].max() < 500_000


def test_cap_outliers():
    df = pd.DataFrame({"pay": [100, 110, 105, 2_000_000]})
    actions = [{
        "id": "1",
        "type": "cap_outliers",
        "column": "pay",
        "lower_percentile": 0,
        "upper_percentile": 75,
    }]
    result = apply_cleaning_actions(df, actions)
    assert result["pay"].max() < 2_000_000


def test_fill_all_nulls_reaches_zero_missing(messy_df):
    result = apply_cleaning_actions(
        messy_df,
        [{"id": "1", "type": "fill_all_nulls"}],
    )
    assert result.isna().sum().sum() == 0
    assert preview_stats(result)["missing_pct"] == 0.0


def test_proposal_fill_all_nulls(messy_df):
    proposal = proposal_fill_all_nulls(messy_df)
    assert len(proposal["actions"]) == 1
    assert proposal["actions"][0]["type"] == "fill_all_nulls"
    cleaned = apply_cleaning_actions(messy_df, proposal["actions"])
    assert cleaned.isna().sum().sum() == 0


def test_per_column_proposal_covers_every_column(messy_df):
    proposal = build_per_column_proposal(messy_df, use_ai=False)
    column_actions = [a for a in proposal["actions"] if a.get("column")]
    assert len(column_actions) == len(messy_df.columns)


def test_per_column_no_action_cap():
    df = pd.DataFrame({f"col_{i}": [None, i, i + 1] for i in range(25)})
    proposal = build_per_column_proposal(df, use_ai=False)
    column_actions = [a for a in proposal["actions"] if a.get("column")]
    assert len(column_actions) == 25


def test_heuristic_fill_null_for_missing(messy_df):
    action = heuristic_column_action(messy_df, "score")
    assert action["type"] == "fill_null"
    assert action["strategy"] in ("median", "mean")


def test_column_profile_has_null_pct(messy_df):
    p = column_profile(messy_df, "score")
    assert p["nulls"] == 1
    assert p["null_pct"] == 25.0


def test_fallback_detects_salary_outlier():
    df = pd.DataFrame({
        "Country": ["Gabon", "USA"],
        "ConvertedCompYearly": [2_000_000, 143_000],
    })
    proposal = _fallback_proposal(df)
    types = {a["type"] for a in proposal["actions"]}
    assert "drop_outlier_rows" in types
