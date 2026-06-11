"""Per-column data-quality analysis and AI/heuristic cleaning recommendations."""

from __future__ import annotations

import json
import re
from typing import Any

import ollama
import pandas as pd

from athena.config import (
    CLEANING_BATCH_SIZE,
    CLEANING_NUM_PREDICT_PER_BATCH,
    MODEL,
)
from athena.llm.schema import coerce_numeric_like_columns, numeric_columns

COLUMN_ACTION_TYPES = frozenset({
    "drop_column",
    "fill_null",
    "drop_outlier_rows",
    "cap_outliers",
    "skip",
})

FILL_STRATEGIES = frozenset({"median", "mean", "mode", "constant"})
OUTLIER_METHODS = frozenset({"iqr", "zscore"})

CLEANING_DECISION_GUIDE = """
You are a senior data engineer. Analyze EACH column independently and recommend exactly ONE action.

## Decision order (apply the first rule that matches)

1. **drop_column** — Column is useless or too broken to salvage:
   - ≥90% null/missing (unless it is clearly an ID/key column with <95% null)
   - Only one distinct non-null value (constant) and not a meaningful label column
   - Duplicate of another column (same values as another field)
   - Meaningless index columns (Unnamed: 0, row numbers)

2. **drop_outlier_rows** — Numeric column has a few extreme values that are clearly errors
   (e.g. salary 2M when median is 90k, age 999, negative counts). Use when:
   - Outliers are <5% of rows AND max is far from median (e.g. >3× median or IQR rule)
   - Removing rows is safer than clipping for obvious bad entries
   - method: "iqr" (factor 1.5) or "zscore" (threshold 3.0)

3. **cap_outliers** — Numeric column has extremes that may be valid but skew charts/averages:
   - Use when you want to KEEP rows but tame tails (compensation, revenue, duration)
   - Prefer cap when outliers are >5% of rows or might be legitimate high earners
   - lower_percentile 1, upper_percentile 99 (or 5/95 for heavy tails)

4. **fill_null** — Column is worth keeping but has missing values:
   - **median**: numeric, skewed, or ordinal (typical default for numbers)
   - **mean**: numeric, roughly symmetric, low skew
   - **mode**: categorical / text / boolean with low cardinality
   - **constant**: high-cardinality IDs, free text, or "Unknown" for survey blanks
     (include "value": "Unknown" or another sensible default)
   - Never fill ID/key columns that should stay null — drop_column instead if mostly empty

5. **skip** — Column is clean: no meaningful nulls, no harmful outliers, not constant junk.

## Row-level (dataset, not per-column — only if mentioned separately)
- **drop_duplicate_rows**: duplicate full rows exist
- **drop_rows_all_null**: many rows are completely empty across key fields

## Do NOT use fill_all_nulls when analyzing columns one-by-one — use fill_null per column.

## ID / key columns
- Names like id, uuid, user_id, respondent_id: prefer **skip** or **fill_null** with constant
  only if a few nulls; **drop_column** only if ≥95% null.
"""

_ID_LIKE = re.compile(
    r"(^id$|_id$|^uuid$|guid$|key$|identifier$|respondent|user_?id)",
    re.I,
)

_JUNK_INDEX_NAME = re.compile(r"^unnamed([:_\s.]|$)", re.I)


def _is_id_like_column(name: str) -> bool:
    return bool(_ID_LIKE.search(name.replace(" ", "_")))


def is_junk_index_column(df: pd.DataFrame, col: str) -> bool:
    """Pandas export artifacts like 'Unnamed: 0' holding a row-number sequence."""
    if not _JUNK_INDEX_NAME.match(str(col).strip()):
        return False
    series = df[col].dropna()
    if len(series) == 0:
        return True
    if not pd.api.types.is_numeric_dtype(series.dtype):
        return False
    # Mostly-unique tolerates duplicated rows in otherwise sequential exports.
    return series.nunique() >= len(series) * 0.95


def duplicate_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map each duplicate column to the earlier column with identical values."""
    out: dict[str, str] = {}
    by_hash: dict[int, list[str]] = {}
    for col in df.columns:
        try:
            h = int(pd.util.hash_pandas_object(df[col], index=False).sum())
        except TypeError:
            continue
        duplicate_of = None
        for prior in by_hash.get(h, []):
            if df[col].equals(df[prior]):
                duplicate_of = prior
                break
        if duplicate_of:
            out[col] = duplicate_of
        else:
            by_hash.setdefault(h, []).append(col)
    return out


def _numeric_skew(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 8:
        return 0.0
    std = s.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(abs(s.skew()))


def _outlier_count(series: pd.Series, factor: float = 1.5) -> int:
    s = series.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    low, high = q1 - factor * iqr, q3 + factor * iqr
    return int(((series < low) | (series > high)).sum())


def column_profile(df: pd.DataFrame, col: str) -> dict[str, Any]:
    n = len(df)
    series = df[col]
    nulls = int(series.isna().sum())
    null_pct = (nulls / n * 100) if n else 0.0
    non_null = series.dropna()
    nunique = int(series.nunique(dropna=True))
    is_numeric = col in numeric_columns(df)
    profile: dict[str, Any] = {
        "column": col,
        "dtype": str(series.dtype),
        "rows": n,
        "nulls": nulls,
        "null_pct": round(null_pct, 2),
        "nunique": nunique,
        "is_numeric": is_numeric,
        "is_id_like": _is_id_like_column(col),
    }
    if is_numeric and len(non_null):
        profile["min"] = float(non_null.min())
        profile["max"] = float(non_null.max())
        profile["median"] = float(non_null.median())
        profile["mean"] = float(non_null.mean())
        profile["skew"] = round(_numeric_skew(series), 2)
        sorted_vals = non_null.sort_values()
        mx = float(sorted_vals.iloc[-1])
        med = profile["median"]
        outlier_rows = _outlier_count(series)
        if outlier_rows == 0 and len(sorted_vals) >= 2:
            second = float(sorted_vals.iloc[-2])
            if med > 0 and mx > med * 3:
                outlier_rows = 1
            elif second > 0 and mx / second > 8:
                outlier_rows = 1
        profile["outlier_rows"] = outlier_rows
        if len(sorted_vals) >= 2:
            profile["second_largest"] = float(sorted_vals.iloc[-2])
    elif len(non_null):
        profile["sample_values"] = non_null.astype(str).head(3).tolist()
    return profile


def format_column_profile_line(p: dict[str, Any]) -> str:
    col = p["column"]
    base = (
        f"- `{col}`: dtype={p['dtype']}, nulls={p['nulls']} ({p['null_pct']}%), "
        f"unique={p['nunique']}"
    )
    if p.get("is_id_like"):
        base += ", id_like=true"
    if p.get("is_numeric"):
        extras = []
        if "min" in p:
            extras.append(f"min={p['min']:g}")
            extras.append(f"max={p['max']:g}")
            extras.append(f"median={p['median']:g}")
        if p.get("outlier_rows"):
            extras.append(f"outlier_rows={p['outlier_rows']}")
        if p.get("skew") is not None:
            extras.append(f"skew={p['skew']}")
        if extras:
            base += ", " + ", ".join(extras)
    elif p.get("sample_values"):
        base += f", samples={p['sample_values']}"
    return base


def heuristic_column_action(df: pd.DataFrame, col: str, profile: dict[str, Any] | None = None) -> dict:
    """Rule-based recommendation for a single column (always returns an action dict)."""
    p = profile or column_profile(df, col)
    n = p["rows"]
    null_pct = p["null_pct"] / 100.0
    nunique = p["nunique"]
    is_id = p.get("is_id_like", False)
    reason_base = f"Column `{col}`"

    if null_pct >= 0.9 and not (is_id and null_pct < 0.95):
        return {
            "id": col,
            "column": col,
            "type": "drop_column",
            "reason": f"{reason_base}: {p['null_pct']}% missing — too empty to keep",
        }

    if n > 0 and nunique <= 1 and null_pct < 0.9:
        return {
            "id": col,
            "column": col,
            "type": "drop_column",
            "reason": f"{reason_base}: only one distinct value",
        }

    if p.get("is_numeric") and p.get("outlier_rows", 0) > 0:
        med = p.get("median") or 0
        mx = p.get("max") or 0
        out_pct = p["outlier_rows"] / n if n else 0
        few_extremes = p["outlier_rows"] <= max(5, int(n * 0.05))
        second = p.get("second_largest")
        obvious_error = med > 0 and mx > med * 3
        if not obvious_error and second and second > 0:
            obvious_error = mx / second > 8
        if few_extremes and obvious_error:
            return {
                "id": col,
                "column": col,
                "type": "drop_outlier_rows",
                "method": "iqr",
                "factor": 1.5,
                "reason": (
                    f"{reason_base}: {p['outlier_rows']} extreme row(s) "
                    f"(max {mx:g} vs median {med:g})"
                ),
            }
        if out_pct <= 0.05 and obvious_error:
            return {
                "id": col,
                "column": col,
                "type": "drop_outlier_rows",
                "method": "iqr",
                "factor": 1.5,
                "reason": (
                    f"{reason_base}: {p['outlier_rows']} extreme row(s) "
                    f"(max {mx:g} vs median {med:g})"
                ),
            }
        return {
            "id": col,
            "column": col,
            "type": "cap_outliers",
            "lower_percentile": 1,
            "upper_percentile": 99,
            "reason": (
                f"{reason_base}: cap outliers (max {mx:g}) to reduce chart skew"
            ),
        }

    if p["nulls"] > 0:
        if p.get("is_numeric"):
            strategy = "median" if (p.get("skew") or 0) > 1 else "mean"
            return {
                "id": col,
                "column": col,
                "type": "fill_null",
                "strategy": strategy,
                "reason": (
                    f"{reason_base}: fill {p['nulls']} missing values with {strategy}"
                ),
            }
        if nunique <= max(20, n * 0.5) if n else 20:
            return {
                "id": col,
                "column": col,
                "type": "fill_null",
                "strategy": "mode",
                "reason": f"{reason_base}: fill {p['nulls']} missing values with mode",
            }
        return {
            "id": col,
            "column": col,
            "type": "fill_null",
            "strategy": "constant",
            "value": "Unknown",
            "reason": (
                f"{reason_base}: high-cardinality text — fill {p['nulls']} with 'Unknown'"
            ),
        }

    return {
        "id": col,
        "column": col,
        "type": "skip",
        "reason": f"{reason_base}: no missing values or outliers detected",
    }


def _valid_action_fields(act: dict) -> bool:
    """Validate strategy/method/percentiles at parse time — invalid actions are
    dropped so the heuristic fallback covers those columns instead."""
    action_type = act.get("type")

    if action_type == "fill_null":
        strategy = act.get("strategy", "median")
        if strategy not in FILL_STRATEGIES:
            return False
        if strategy == "constant" and "value" not in act:
            return False
        return True

    if action_type == "drop_outlier_rows":
        return act.get("method", "iqr") in OUTLIER_METHODS

    if action_type == "cap_outliers":
        lo = act.get("lower_percentile", 1)
        hi = act.get("upper_percentile", 99)
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            return False
        return 0 <= lo < hi <= 100

    return True


def _parse_batch_actions(raw: str, expected_columns: list[str]) -> list[dict]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

    if not isinstance(data, dict):
        return []
    actions_raw = data.get("actions", [])
    if not isinstance(actions_raw, list):
        return []

    expected = set(expected_columns)
    parsed: list[dict] = []
    seen: set[str] = set()

    for i, act in enumerate(actions_raw):
        if not isinstance(act, dict):
            continue
        action_type = act.get("type")
        if action_type not in COLUMN_ACTION_TYPES:
            continue
        col = act.get("column")
        if not col or col not in expected or col in seen:
            continue
        if not _valid_action_fields(act):
            continue
        seen.add(col)
        aid = str(act.get("id", col))
        parsed.append({**act, "id": aid, "type": action_type, "column": col})

    return parsed


def _build_batch_prompt(
    df: pd.DataFrame,
    batch_cols: list[str],
    profiles: dict[str, dict],
    batch_index: int,
    batch_total: int,
) -> str:
    lines = [format_column_profile_line(profiles[c]) for c in batch_cols]
    col_block = "\n".join(lines)
    return f"""{CLEANING_DECISION_GUIDE}

Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns.
Batch {batch_index + 1} of {batch_total} — analyze EVERY column below (one action each).

Columns in this batch:
{col_block}

Return ONLY valid JSON:
{{
  "actions": [
    {{
      "id": "exact_column_name",
      "column": "exact_column_name",
      "type": "fill_null",
      "strategy": "median",
      "reason": "short reason"
    }},
    {{
      "id": "other_column",
      "column": "other_column",
      "type": "skip",
      "reason": "no issues"
    }}
  ]
}}

Requirements:
- Include exactly one action object per column listed above ({len(batch_cols)} actions).
- "id" and "column" must match the exact column name.
- Allowed types: drop_column, fill_null, drop_outlier_rows, cap_outliers, skip
- fill_null strategy: median, mean, mode, or constant (with "value" if constant)
- Do not omit any column from this batch
"""


def _run_ai_batch(
    df: pd.DataFrame,
    batch_cols: list[str],
    profiles: dict[str, dict],
    batch_index: int,
    batch_total: int,
) -> list[dict] | None:
    """One Ollama call for a column batch. None when the call itself fails."""
    prompt = _build_batch_prompt(df, batch_cols, profiles, batch_index, batch_total)
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,
                "num_predict": CLEANING_NUM_PREDICT_PER_BATCH,
            },
        )
    except Exception:
        return None
    raw = response["message"]["content"].strip()
    return _parse_batch_actions(raw, batch_cols)


def _ai_analyze_column_batches(
    df: pd.DataFrame,
    profiles: dict[str, dict],
    columns: list[str] | None = None,
) -> dict:
    """
    Batched AI analysis with retries.
    Failed calls are retried once; partially answered batches get one
    smaller follow-up batch for the missing columns.
    Returns {"actions": [...], "failed_batches": int, "total_batches": int}.
    """
    columns = list(df.columns) if columns is None else columns
    if not columns:
        return {"actions": [], "failed_batches": 0, "total_batches": 0}

    batches = [
        columns[i : i + CLEANING_BATCH_SIZE]
        for i in range(0, len(columns), CLEANING_BATCH_SIZE)
    ]
    all_actions: list[dict] = []
    failed = 0

    for idx, batch_cols in enumerate(batches):
        actions = _run_ai_batch(df, batch_cols, profiles, idx, len(batches))
        if actions is None:
            actions = _run_ai_batch(df, batch_cols, profiles, idx, len(batches))
        if actions is None:
            failed += 1
            continue

        covered = {a["column"] for a in actions}
        missing = [c for c in batch_cols if c not in covered]
        if missing and len(missing) < len(batch_cols):
            extra = _run_ai_batch(df, missing, profiles, idx, len(batches))
            if extra:
                actions.extend(extra)

        all_actions.extend(actions)

    return {
        "actions": all_actions,
        "failed_batches": failed,
        "total_batches": len(batches),
    }


def dataset_level_actions(df: pd.DataFrame) -> list[dict]:
    actions: list[dict] = []
    dupes = int(df.duplicated().sum())
    if dupes > 0:
        actions.append({
            "id": "_drop_duplicates",
            "type": "drop_duplicate_rows",
            "reason": f"Remove {dupes} fully duplicate rows",
        })

    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows > 0:
        actions.append({
            "id": "_drop_empty_rows",
            "type": "drop_rows_all_null",
            "reason": f"Drop {empty_rows} rows that are entirely empty",
        })
    return actions


def build_per_column_proposal(
    df: pd.DataFrame,
    *,
    use_ai: bool = True,
) -> dict:
    """
    Analyze every column and return one recommendation per column (no action cap).
    Uses batched AI when use_ai=True; fills gaps with heuristics.
    Each column action carries "source": "ai" or "heuristic".
    """
    # Parse numeric-like text first so profiles and outlier checks see real numbers.
    df = coerce_numeric_like_columns(df)

    profiles = {col: column_profile(df, col) for col in df.columns}
    actions: list[dict] = dataset_level_actions(df)

    # Deterministic structural rules take precedence — no AI needed.
    forced: dict[str, dict] = {}
    dupes = duplicate_columns(df)
    for col in df.columns:
        if is_junk_index_column(df, col):
            forced[col] = {
                "id": col,
                "column": col,
                "type": "drop_column",
                "source": "heuristic",
                "reason": f"Column `{col}`: junk index column (row numbers from export)",
            }
        elif col in dupes:
            forced[col] = {
                "id": col,
                "column": col,
                "type": "drop_column",
                "source": "heuristic",
                "reason": f"Column `{col}`: exact duplicate of `{dupes[col]}`",
            }

    ai_by_col: dict[str, dict] = {}
    failed_batches = 0
    total_batches = 0
    if use_ai and df.shape[1] > 0:
        ai_cols = [c for c in df.columns if c not in forced]
        outcome = _ai_analyze_column_batches(df, profiles, ai_cols)
        failed_batches = outcome["failed_batches"]
        total_batches = outcome["total_batches"]
        for act in outcome["actions"]:
            ai_by_col[act["column"]] = {**act, "source": "ai"}

    issue_count = 0
    for col in df.columns:
        if col in forced:
            action = forced[col]
        elif col in ai_by_col:
            action = ai_by_col[col]
        else:
            action = {**heuristic_column_action(df, col, profiles[col]), "source": "heuristic"}
        if action.get("type") != "skip":
            issue_count += 1
        actions.append(action)

    n_cols = len(df.columns)
    ai_count = len(ai_by_col)
    summary = (
        f"Reviewed all {n_cols} column(s): {issue_count} need changes, "
        f"{n_cols - issue_count} look clean."
    )
    if use_ai:
        summary += f" AI analyzed {ai_count} column(s) in batches; heuristics cover the rest."

    return {
        "summary": summary,
        "actions": actions,
        "ai_columns": ai_count,
        "ai_failed_batches": failed_batches,
        "ai_total_batches": total_batches,
    }
