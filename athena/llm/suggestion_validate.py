import re

import pandas as pd

from athena.llm.schema import comparable_numeric_columns, numeric_columns

# (phrases in question, substrings to find in column names)
_TOPIC_HINTS: list[tuple[list[str], list[str]]] = [
    (
        ["work experience", "years of work", "years of experience", "professional experience", "tenure"],
        ["workexp", "yearscode", "yearsexp", "yearscodepro", "yearsincode", "yearsexperience"],
    ),
    (
        ["salary", "compensation", "pay", "income", "earnings"],
        ["comp", "salary", "pay", "income", "wage"],
    ),
    (
        ["country", "region", "location"],
        ["country", "region", "location", "city"],
    ),
    (
        ["remote", "work from home"],
        ["remote", "wfh"],
    ),
    (
        ["education", "degree", "learn", "bootcamp"],
        ["educ", "learn", "degree", "school"],
    ),
    (
        ["activity", "activities", "coding", "language", "skill"],
        ["activ", "code", "lang", "skill", "tech"],
    ),
]

_GENERIC_OK = (
    "how many rows",
    "how many records",
    "missing values",
    "missing cells",
    "duplicate rows",
    "columns have",
    "in the dataset",
)

_COMPARE_PATTERN = re.compile(
    r"(?:greater than|more than|less than|at least|over|under)\s+(\d+)|>\s*(\d+)|<\s*(\d+)"
)


def columns_named_in_question(question: str, df: pd.DataFrame) -> list[str]:
    q = question.lower()
    found = [c for c in df.columns if c.lower() in q]
    for match in re.findall(r"`([^`]+)`", question):
        if match in df.columns and match not in found:
            found.append(match)
    return found


def columns_for_topic(question: str, df: pd.DataFrame) -> list[str]:
    q = question.lower()
    matched_cols: list[str] = []
    for phrases, col_hints in _TOPIC_HINTS:
        if not any(p in q for p in phrases):
            continue
        for col in df.columns:
            cl = col.lower().replace(" ", "").replace("_", "")
            if any(h in cl for h in col_hints if len(h) >= 5):
                matched_cols.append(col)
    return matched_cols


def question_needs_numeric_compare(question: str) -> bool:
    return _COMPARE_PATTERN.search(question.lower()) is not None


def _best_comparable_for_topic(question: str, df: pd.DataFrame) -> str | None:
    topic = columns_for_topic(question, df)
    comparable = set(comparable_numeric_columns(df))
    for col in topic:
        if col in comparable:
            return col
    return None


def is_suggestion_answerable(question: str, df: pd.DataFrame) -> bool:
    """True if the dataset has columns that can support this question."""
    named = columns_named_in_question(question, df)
    if named:
        if question_needs_numeric_compare(question):
            return any(c in comparable_numeric_columns(df) for c in named)
        return True
    q = question.lower()
    if any(p in q for p in _GENERIC_OK):
        return True
    if question_needs_numeric_compare(question):
        return _best_comparable_for_topic(question, df) is not None
    topic_cols = columns_for_topic(question, df)
    if not topic_cols:
        return False
    return True


def repair_suggestion(question: str, df: pd.DataFrame) -> str | None:
    """Rewrite a vague question to cite a real column, or return None if impossible."""
    q = question.lower()
    num_match = _COMPARE_PATTERN.search(q)
    threshold = None
    if num_match:
        threshold = next(g for g in num_match.groups() if g)

    if question_needs_numeric_compare(question):
        col = _best_comparable_for_topic(question, df)
        if not col and columns_named_in_question(question, df):
            named = columns_named_in_question(question, df)
            col = next((c for c in named if c in comparable_numeric_columns(df)), None)
        if not col:
            return None
        if threshold:
            return f"How many rows have `{col}` greater than {threshold}?"
        return f"How many rows have a valid numeric `{col}` value?"

    named = columns_named_in_question(question, df)
    if named:
        col = named[0]
        if "how many" in q and col in comparable_numeric_columns(df):
            return f"What is the count of rows where `{col}` is not missing?"
        if any(w in q for w in ("average", "mean", "median")) and col in comparable_numeric_columns(df):
            return f"What is the median of `{col}` (after pd.to_numeric if needed)?"
        if "top" in q or "most common" in q:
            return f"What are the top 10 most common values in `{col}`?"
        return question

    topic_cols = columns_for_topic(question, df)
    if not topic_cols:
        return None

    col = topic_cols[0]
    if "how many" in q and col in comparable_numeric_columns(df):
        return f"What is the count of rows where `{col}` is not missing?"

    if any(w in q for w in ("average", "mean", "median")) and col in comparable_numeric_columns(df):
        return f"What is the median of `{col}`?"

    if "top" in q or "most common" in q:
        return f"What are the top 10 most common values in `{col}`?"

    return f"What are the top 10 most common values in `{col}`?"


def normalize_suggestion(question: str, df: pd.DataFrame) -> str | None:
    """Ensure the question cites a real column; drop if no mapping exists."""
    named = columns_named_in_question(question, df)
    if named and question_needs_numeric_compare(question):
        if not any(c in comparable_numeric_columns(df) for c in named):
            return repair_suggestion(question, df)
        if not re.search(r"`[^`]+`", question):
            col = next(c for c in named if c in comparable_numeric_columns(df))
            num_match = _COMPARE_PATTERN.search(question.lower())
            if num_match:
                threshold = next(g for g in num_match.groups() if g)
                return f"How many rows have `{col}` greater than {threshold}?"
    if columns_named_in_question(question, df) and not question_needs_numeric_compare(question):
        return question
    if not is_suggestion_answerable(question, df):
        return repair_suggestion(question, df)
    return repair_suggestion(question, df) or question


def filter_answerable_suggestions(
    questions: list[str],
    df: pd.DataFrame,
    n: int,
) -> list[str]:
    """Keep only answerable questions; repair or drop the rest."""
    out: list[str] = []
    seen: set[str] = set()

    for q in questions:
        candidate = normalize_suggestion(q, df)
        if not candidate:
            continue
        key = candidate.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= n:
            break
    return out
