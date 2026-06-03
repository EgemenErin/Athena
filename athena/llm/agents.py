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


def review_result(
    question: str,
    plan: dict | None,
    result,
    code: str | None,
) -> dict:
    """
    Reviewer agent: check if result matches the question/plan.
    Returns {"ok": bool, "feedback": str}.
    """
    if result is None:
        return {"ok": False, "feedback": "No result was produced."}

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

    return {"ok": True, "feedback": ""}


def build_analyst_user_message(question: str, plan: dict | None) -> str:
    if plan:
        return f"{format_plan_for_analyst(plan)}\n\nUser question: {question}"
    return question
