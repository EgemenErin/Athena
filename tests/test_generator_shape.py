import pandas as pd

from athena.llm.agents import (
    check_expected_shape,
    deterministic_review,
)


def _corr_matrix():
    return pd.DataFrame(
        [[1.0, 0.1], [0.1, 1.0]],
        columns=["Salary", "Age"],
        index=["Salary", "Age"],
    )


def _grouped_table():
    return pd.DataFrame({
        "Country": ["USA", "Germany"],
        "mean_salary": [120000.0, 85000.0],
    })


# ---------- check_expected_shape ----------

def test_scalar_plan_rejects_big_table():
    plan = {"expected_shape": "scalar"}
    feedback = check_expected_shape(plan, _grouped_table())
    assert feedback is not None
    assert "scalar" in feedback or "single number" in feedback


def test_scalar_plan_accepts_scalar():
    plan = {"expected_shape": "scalar"}
    assert check_expected_shape(plan, 42) is None


def test_scalar_plan_rejects_correlation():
    plan = {"expected_shape": "scalar"}
    assert check_expected_shape(plan, _corr_matrix()) is not None


def test_correlation_plan_rejects_grouped_table():
    plan = {"expected_shape": "correlation"}
    assert check_expected_shape(plan, _grouped_table()) is not None


def test_correlation_plan_accepts_matrix():
    plan = {"expected_shape": "correlation"}
    assert check_expected_shape(plan, _corr_matrix()) is None


def test_grouped_table_plan_rejects_scalar():
    plan = {"expected_shape": "grouped_table"}
    assert check_expected_shape(plan, 42) is not None


def test_grouped_table_plan_rejects_correlation_matrix():
    plan = {"expected_shape": "grouped_table"}
    assert check_expected_shape(plan, _corr_matrix()) is not None


def test_no_plan_passes():
    assert check_expected_shape(None, _grouped_table()) is None
    assert check_expected_shape({}, 42) is None


# ---------- deterministic_review ----------

def test_count_question_never_gets_correlation():
    verdict = deterministic_review("How many developers use Python?", None, _corr_matrix())
    assert verdict["ok"] is False
    assert "correlation" in verdict["feedback"].lower()


def test_correlation_question_allows_matrix():
    verdict = deterministic_review(
        "What is the correlation between salary and age?", None, _corr_matrix()
    )
    assert verdict["ok"] is True


def test_empty_result_fails():
    verdict = deterministic_review("Top countries", None, pd.DataFrame({"a": []}))
    assert verdict["ok"] is False
    assert "empty" in verdict["feedback"].lower()


def test_nan_scalar_fails():
    # Exact-match filter hit zero rows → .mean() returned NaN.
    import numpy as np

    verdict = deterministic_review(
        "Average compensation for developers in the United States",
        {"columns": ["Country", "ConvertedCompYearly"], "expected_shape": "scalar"},
        float("nan"),
    )
    assert verdict["ok"] is False
    assert "contains" in verdict["feedback"]

    verdict_np = deterministic_review("Average pay", None, np.float64("nan"))
    assert verdict_np["ok"] is False


def test_all_nan_series_fails():
    s = pd.Series([float("nan"), float("nan")], index=["a", "b"])
    verdict = deterministic_review("Average pay by country", None, s)
    assert verdict["ok"] is False


def test_all_nan_dataframe_fails():
    df = pd.DataFrame({"Country": ["US", "DE"], "pay": [float("nan")] * 2})
    # Only entirely-NaN frames fail; partial NaN columns are fine.
    partial = deterministic_review("pay by country", None, df)
    assert partial["ok"] is True
    all_nan = pd.DataFrame({"a": [float("nan")], "b": [float("nan")]})
    verdict = deterministic_review("pay by country", None, all_nan)
    assert verdict["ok"] is False


def test_valid_scalar_zero_passes():
    # 0 is a legitimate answer, not an empty result.
    verdict = deterministic_review("How many rows have pay over 1M?", None, 0)
    assert verdict["ok"] is True


def test_string_result_passes():
    verdict = deterministic_review("Which country is most common?", None, "Germany")
    assert verdict["ok"] is True


def test_unrelated_columns_fail():
    plan = {"columns": ["Country", "Salary"]}
    result = pd.DataFrame({"FavoriteColor": ["red"], "ShoeSize": [42]})
    verdict = deterministic_review("Average salary by country", plan, result)
    assert verdict["ok"] is False


def test_related_columns_pass():
    plan = {"columns": ["Country", "Salary"]}
    result = pd.DataFrame({"Country": ["USA"], "mean_salary": [120000.0]})
    verdict = deterministic_review("Average salary by country", plan, result)
    assert verdict["ok"] is True


def test_generic_agg_columns_pass():
    plan = {"columns": ["LearnCode"]}
    result = pd.DataFrame({"index": ["Books"], "count": [12]})
    verdict = deterministic_review("Most common learning methods", plan, result)
    assert verdict["ok"] is True


def test_scalar_result_with_plan_columns_passes():
    plan = {"columns": ["Salary"], "expected_shape": "scalar"}
    verdict = deterministic_review("How many rows have salary above 100k?", plan, 17)
    assert verdict["ok"] is True
