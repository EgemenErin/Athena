import json
import re

import ollama
import pandas as pd

from athena.config import MODEL
from athena.llm.schema import build_column_index
from athena.llm.suggestion_validate import columns_for_topic, columns_named_in_question
def _extract_json(raw: str) -> dict | None:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _sanitize_plan_columns(plan: dict, df: pd.DataFrame, question: str) -> dict:
    """Keep only columns that exist; infer from question if planner picked wrong ones."""
    valid = [c for c in (plan.get("columns") or []) if c in df.columns]
    if not valid:
        valid = columns_named_in_question(question, df)
    if not valid:
        valid = columns_for_topic(question, df)
    plan = {**plan, "columns": valid[:6]}
    return plan


def plan_analysis(question: str, schema: str, df: pd.DataFrame | None = None) -> dict | None:
    """Planner agent: interpret question and define expected result shape."""
    schema_excerpt = schema if len(schema) <= 3500 else schema[:3500] + "\n... (truncated)"
    column_index = build_column_index(df) if df is not None else ""
    index_block = f"\n{column_index}\n" if column_index else ""
    prompt = f"""You are an analytics planner. Read the user question and schema, then output ONLY JSON:

{{
  "intent": "one sentence paraphrase of what to compute",
  "columns": ["exact", "column", "names"],
  "filters": "plain English filter on rows, or empty string",
  "expected_shape": "scalar | list | grouped_table | correlation",
  "hints": "pandas hints: explode semicolons, str.contains, groupby, nunique, etc."
}}

Rules:
- columns must exist in the schema
- expected_shape is scalar for counts/single numbers, list for distinct values, grouped_table for top-N or breakdowns, correlation ONLY if user asks correlation
- Do not plan correlation for "how many" or "different activities" questions

{index_block}
Schema:
{schema_excerpt}

User question: {question}
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2, "num_predict": 400},
        )
        data = _extract_json(response["message"]["content"].strip())
        if isinstance(data, dict) and data.get("intent"):
            if df is not None:
                return _sanitize_plan_columns(data, df, question)
            return data
    except Exception:
        pass
    return None


def format_plan_for_analyst(plan: dict) -> str:
    cols = ", ".join(plan.get("columns") or []) or "see schema"
    return (
        f"Analysis plan (follow this closely):\n"
        f"- Intent: {plan.get('intent', '')}\n"
        f"- Columns: {cols}\n"
        f"- Filters: {plan.get('filters') or 'none'}\n"
        f"- Expected result: {plan.get('expected_shape', 'grouped_table')}\n"
        f"- Hints: {plan.get('hints') or 'none'}\n"
    )


_COUNT_QUESTION = re.compile(r"\b(how many|count|number of)\b", re.IGNORECASE)
_CORRELATION_QUESTION = re.compile(r"\b(correlat|relationship between)\b", re.IGNORECASE)

# Result column names that never need to match plan columns (aggregation outputs).
_GENERIC_RESULT_COLUMNS = frozenset({
    "count", "counts", "value", "values", "mean", "median", "sum", "total",
    "n", "pct", "percent", "percentage", "share", "result", "index", "size",
    "min", "max", "std", "avg", "average", "frequency", "freq",
})


def _is_correlation_like(result) -> bool:
    if not isinstance(result, pd.DataFrame):
        return False
    rows, cols = result.shape
    if rows != cols or rows < 2 or rows > 12:
        return False
    return len(result.select_dtypes(include="number").columns) == cols


def _result_is_empty(result) -> bool:
    """Empty frames/series, all-NaN results, and NaN scalars all count as 'no answer'."""
    if result is None:
        return True
    if isinstance(result, pd.DataFrame):
        if len(result) == 0:
            return True
        try:
            return bool(result.isna().all().all())
        except (TypeError, ValueError):
            return False
    if isinstance(result, pd.Series):
        if len(result) == 0:
            return True
        try:
            return bool(result.isna().all())
        except (TypeError, ValueError):
            return False
    try:
        return bool(pd.isna(result))
    except (TypeError, ValueError):
        return False


def _normalize_name(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _columns_relate(result_col: str, plan_cols: list[str]) -> bool:
    rc = _normalize_name(result_col)
    if not rc or rc in {_normalize_name(g) for g in _GENERIC_RESULT_COLUMNS}:
        return True
    for plan_col in plan_cols:
        pc = _normalize_name(plan_col)
        if pc and (pc in rc or rc in pc):
            return True
    return False


def check_expected_shape(plan: dict | None, result) -> str | None:
    """Return feedback when the result shape contradicts the plan, else None."""
    shape = (plan or {}).get("expected_shape")
    if not shape:
        return None

    if shape == "scalar":
        if isinstance(result, pd.DataFrame):
            if result.shape[0] > 1 or result.shape[1] > 2:
                return (
                    "The plan expected a single number (scalar), but the result is a "
                    f"{result.shape[0]}×{result.shape[1]} table. Return one value "
                    "(e.g. a count or aggregate) assigned to `result`."
                )
            if _is_correlation_like(result):
                return (
                    "The plan expected a single number, but the result is a correlation "
                    "matrix. Compute the requested count/aggregate instead."
                )
        elif isinstance(result, pd.Series) and len(result) > 1:
            return (
                "The plan expected a single number (scalar), but the result is a series "
                f"with {len(result)} values. Return one value assigned to `result`."
            )
        return None

    if shape == "correlation":
        if not _is_correlation_like(result):
            return (
                "The plan expected a correlation matrix, but the result is not a square "
                "numeric matrix. Use df[cols].corr() on the relevant numeric columns."
            )
        return None

    if shape in ("grouped_table", "list"):
        if not isinstance(result, (pd.DataFrame, pd.Series)):
            return (
                f"The plan expected a {shape.replace('_', ' ')}, but the result is a "
                f"single {type(result).__name__}. Return a table/series with one row "
                "per group or value."
            )
        if _is_correlation_like(result):
            return (
                f"The plan expected a {shape.replace('_', ' ')}, but the result looks "
                "like a correlation matrix. Use groupby/value_counts instead of .corr()."
            )
    return None


def deterministic_review(
    question: str,
    plan: dict | None,
    result,
) -> dict:
    """
    Rule-based result checks that run without the LLM.
    Returns {"ok": bool, "feedback": str}.
    """
    if _result_is_empty(result):
        return {
            "ok": False,
            "feedback": (
                "The result is empty or NaN — the filter probably matched zero rows. "
                "The value in the question may not match the data exactly (e.g. "
                "'United States' may be stored as 'United States of America'). "
                "Use .str.contains(<distinctive substring>, case=False, na=False) "
                "instead of ==, and check the schema sample values for exact spellings."
            ),
        }

    if (
        _COUNT_QUESTION.search(question)
        and not _CORRELATION_QUESTION.search(question)
        and _is_correlation_like(result)
    ):
        return {
            "ok": False,
            "feedback": (
                "The user asked for a count, but the result is a correlation matrix. "
                "Compute the count (len, sum, nunique, or value_counts) instead of .corr()."
            ),
        }

    shape_feedback = check_expected_shape(plan, result)
    if shape_feedback:
        return {"ok": False, "feedback": shape_feedback}

    plan_cols = [c for c in ((plan or {}).get("columns") or []) if c]
    if plan_cols and isinstance(result, pd.DataFrame) and len(result.columns) > 0:
        names = list(result.columns) + [n for n in (result.index.names or []) if n]
        if not any(_columns_relate(name, plan_cols) for name in names):
            return {
                "ok": False,
                "feedback": (
                    f"The result columns {list(result.columns)[:4]} do not relate to the "
                    f"planned columns {plan_cols[:4]}. Use the planned columns from the schema."
                ),
            }

    return {"ok": True, "feedback": ""}


def review_result(
    question: str,
    plan: dict | None,
    result,
    code: str | None,
) -> dict:
    """
    Reviewer: deterministic rule checks first, then LLM judgment.
    Fails closed — if the LLM reply cannot be parsed, the rule-based
    verdict stands instead of silently passing.
    Returns {"ok": bool, "feedback": str}.
    """
    if result is None:
        return {"ok": False, "feedback": "No result was produced."}

    rule_verdict = deterministic_review(question, plan, result)
    if not rule_verdict["ok"]:
        return rule_verdict

    if isinstance(result, pd.DataFrame):
        preview = result.head(8).to_string(index=False)
        shape = f"{result.shape[0]} rows × {result.shape[1]} cols"
        rows, cols = result.shape
        if rows == cols and rows <= 6 and len(result.select_dtypes(include="number").columns) == cols:
            shape += " (square numeric matrix)"
    elif isinstance(result, pd.Series):
        preview = result.head(8).to_string()
        shape = f"series, {len(result)} values"
    else:
        preview = str(result)
        shape = f"scalar: {type(result).__name__}"

    plan_text = json.dumps(plan, indent=2) if plan else "none"
    code_excerpt = (code or "")[:600]

    prompt = f"""You are a senior analytics reviewer. Decide if the query result answers the user's question.

User question: {question}

Plan:
{plan_text}

Result shape: {shape}
Result preview:
{preview}

Code excerpt:
{code_excerpt}

Reply with ONLY JSON:
{{"ok": true}} 
or 
{{"ok": false, "feedback": "what is wrong and what code should do instead"}}

Fail if: user asked for a count but got correlation matrix; unrelated columns; empty or wrong granularity.
Pass if: result reasonably answers the question.
"""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 300},
        )
        data = _extract_json(response["message"]["content"].strip())
        if isinstance(data, dict) and "ok" in data:
            return {
                "ok": bool(data["ok"]),
                "feedback": str(data.get("feedback", "")),
            }
    except Exception:
        pass

    # LLM reply unusable — fall back to the rule-based verdict (already ok here).
    return rule_verdict


def build_analyst_user_message(question: str, plan: dict | None) -> str:
    if plan:
        return f"{format_plan_for_analyst(plan)}\n\nUser question: {question}"
    return question
