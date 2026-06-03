import pandas as pd

from athena.llm.suggestion_validate import (
    filter_answerable_suggestions,
    is_suggestion_answerable,
    repair_suggestion,
)


def test_experience_question_repaired_with_string_numeric_column():
    df = pd.DataFrame({
        "Country": ["USA", "DE"],
        "YearsCodePro": ["12", "5"],
        "ConvertedCompYearly": [100000, 80000],
        "Age": ["30", "25"],
    })
    q = "How many developers have more than 10 years of work experience?"
    assert is_suggestion_answerable(q, df) is True
    repaired = repair_suggestion(q, df)
    assert repaired is not None
    assert "YearsCodePro" in repaired or "years" in repaired.lower()


def test_hallucinated_topic_rejected():
    df = pd.DataFrame({
        "Country": ["USA"],
        "ConvertedCompYearly": [100000],
    })
    q = "How many developers have more than 10 years of work experience?"
    assert is_suggestion_answerable(q, df) is False
    assert repair_suggestion(q, df) is None


def test_named_column_passes():
    df = pd.DataFrame({"YearsCodePro": [1, 2, 15]})
    q = "How many rows have `YearsCodePro` greater than 10?"
    assert is_suggestion_answerable(q, df) is True


def test_filter_drops_bad_keeps_good():
    df = pd.DataFrame({
        "YearsCodePro": [1, 12, 15],
        "Country": ["USA", "DE", "FR"],
    })
    raw = [
        "How many developers have more than 10 years of work experience?",
        "How many rows have `YearsCodePro` greater than 10?",
        "What are the top 5 values in `Country`?",
    ]
    out = filter_answerable_suggestions(raw, df, 3)
    assert len(out) >= 2
    assert any("YearsCodePro" in q for q in out)
    assert not any(
        "work experience" in q.lower() and "YearsCodePro" not in q and "`" not in q
        for q in out
    )
