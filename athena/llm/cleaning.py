import json
import re

import pandas as pd

from athena.llm.cleaning_columns import (
    FILL_STRATEGIES,
    OUTLIER_METHODS,
    build_per_column_proposal,
)
from athena.llm.schema import build_schema_string, numeric_columns

SUPPORTED_TYPES = frozenset({
    "drop_column",
    "fill_null",
    "fill_all_nulls",
    "drop_duplicate_rows",
    "drop_rows_all_null",
    "drop_outlier_rows",
    "cap_outliers",
    "skip",
})


def _iqr_outlier_mask(series: pd.Series, factor: float = 1.5) -> pd.Series:
    s = series.dropna()
    if len(s) < 4:
        return pd.Series(False, index=series.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    low, high = q1 - factor * iqr, q3 + factor * iqr
    return (series < low) | (series > high)


def _zscore_outlier_mask(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    s = series.dropna()
    if len(s) < 3:
        return pd.Series(False, index=series.index)
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=series.index)
    z = (series - s.mean()).abs() / std
    return z > threshold


def outlier_mask(series: pd.Series, method: str = "iqr", factor: float = 1.5, threshold: float = 3.0) -> pd.Series:
    if method == "zscore":
        return _zscore_outlier_mask(series, threshold)
    return _iqr_outlier_mask(series, factor)


def _outlier_report(df: pd.DataFrame) -> str:
    lines = ["Outlier scan (numeric columns, IQR 1.5×):"]
    n = len(df)
    if n == 0:
        return "\n".join(lines)

    for col in numeric_columns(df):
        s = df[col]
        mask = _iqr_outlier_mask(s)
        count = int(mask.sum())
        if count == 0:
            continue
        med = s.median()
        mx = s.max()
        examples = df.loc[mask, col].head(3).tolist()
        lines.append(
            f"  '{col}': {count} outlier row(s), median={med:g}, max={mx:g}, examples={examples}"
        )

    if len(lines) == 1:
        lines.append("  (none detected via IQR)")
    return "\n".join(lines)


def _dataset_stats(df: pd.DataFrame) -> str:
    lines = [
        f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        f"Total missing cells: {int(df.isna().sum().sum())} ({df.isna().sum().sum() / max(df.size, 1) * 100:.1f}%)",
        "",
        "Per column:",
    ]
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        pct = nulls / len(df) * 100 if len(df) else 0
        nunique = df[col].nunique(dropna=True)
        extra = ""
        if col in numeric_columns(df):
            s = df[col].dropna()
            if len(s):
                extra = f", min={s.min():g}, max={s.max():g}, median={s.median():g}"
        lines.append(
            f"  '{col}': dtype={df[col].dtype}, nulls={nulls} ({pct:.1f}%), unique={nunique}{extra}"
        )
    lines.append("")
    lines.append(_outlier_report(df))
    return "\n".join(lines)


def _parse_cleaning_proposal(raw: str) -> dict | None:
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
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    if "actions" not in data or not isinstance(data["actions"], list):
        return None

    actions = []
    seen_ids: set[str] = set()
    for i, act in enumerate(data["actions"]):
        if not isinstance(act, dict):
            continue
        action_type = act.get("type")
        if action_type not in SUPPORTED_TYPES:
            continue
        aid = str(act.get("id", i + 1))
        if aid in seen_ids:
            continue
        seen_ids.add(aid)
        actions.append({**act, "id": aid, "type": action_type})

    return {
        "summary": str(data.get("summary", "Proposed cleaning actions for your dataset.")),
        "actions": actions,
    }


def _fallback_proposal(df: pd.DataFrame) -> dict:
    """Per-column heuristics only (no LLM). One recommendation per column, no cap."""
    return build_per_column_proposal(df, use_ai=False)


def analyze_for_cleaning(df: pd.DataFrame, schema: str) -> dict:
    """
    Analyze every column (batched AI + heuristics for gaps).
    Returns one action per column plus optional dataset-level actions — no action limit.
    """
    del schema  # column profiles are computed from df; schema kept for API compatibility
    return build_per_column_proposal(df, use_ai=True)


def validate_action(df: pd.DataFrame, action: dict) -> str | None:
    """Return an error message if the action is invalid, else None."""
    action_type = action.get("type")
    if action_type not in SUPPORTED_TYPES:
        return f"Unsupported action type: {action_type}"

    if action_type == "drop_column":
        col = action.get("column")
        if not col or col not in df.columns:
            return f"Unknown column: {col!r}"
        return None

    if action_type in ("fill_all_nulls", "skip"):
        return None

    if action_type == "fill_null":
        col = action.get("column")
        if not col or col not in df.columns:
            return f"Unknown column: {col!r}"
        strategy = action.get("strategy", "median")
        if strategy not in FILL_STRATEGIES:
            return f"Invalid fill strategy: {strategy!r}"
        if strategy in ("mean", "median") and col not in numeric_columns(df):
            return f"Cannot fill {col!r} with {strategy} — column is not numeric"
        if strategy == "constant" and "value" not in action:
            return "constant fill requires a 'value'"
        return None

    if action_type == "drop_rows_all_null":
        cols = action.get("columns")
        if cols is not None:
            if not isinstance(cols, list):
                return "columns must be a list"
            for c in cols:
                if c not in df.columns:
                    return f"Unknown column: {c!r}"
        return None

    if action_type == "drop_outlier_rows":
        col = action.get("column")
        if not col or col not in df.columns:
            return f"Unknown column: {col!r}"
        if col not in numeric_columns(df):
            return f"Column {col!r} is not numeric"
        method = action.get("method", "iqr")
        if method not in OUTLIER_METHODS:
            return f"Invalid outlier method: {method!r}"
        return None

    if action_type == "cap_outliers":
        col = action.get("column")
        if not col or col not in df.columns:
            return f"Unknown column: {col!r}"
        if col not in numeric_columns(df):
            return f"Column {col!r} is not numeric"
        lo = action.get("lower_percentile", 1)
        hi = action.get("upper_percentile", 99)
        if not (0 <= lo < hi <= 100):
            return "percentiles must satisfy 0 <= lower < upper <= 100"
        return None

    return None


def _fill_column(
    series: pd.Series,
    strategy: str,
    value=None,
    *,
    numeric_fallback: float = 0,
    categorical_fallback: str = "Unknown",
) -> pd.Series:
    if strategy == "median":
        fill_val = series.median()
    elif strategy == "mean":
        fill_val = series.mean()
    elif strategy == "mode":
        mode = series.mode()
        fill_val = mode.iloc[0] if len(mode) else None
    elif strategy == "constant":
        fill_val = value
    else:
        fill_val = None

    filled = series.fillna(fill_val)
    if filled.isna().any():
        if pd.api.types.is_numeric_dtype(series):
            filled = filled.fillna(numeric_fallback)
        else:
            filled = filled.fillna(categorical_fallback if value is None else value)
    return filled


def _fill_all_nulls(
    df: pd.DataFrame,
    *,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    categorical_fallback: str = "Unknown",
    numeric_fallback: float = 0,
) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not out[col].isna().any():
            continue
        strategy = (
            numeric_strategy
            if col in numeric_columns(out)
            else categorical_strategy
        )
        out[col] = _fill_column(
            out[col],
            strategy,
            categorical_fallback if strategy == "constant" else None,
            numeric_fallback=numeric_fallback,
            categorical_fallback=categorical_fallback,
        )
    return out


def proposal_fill_all_nulls(df: pd.DataFrame) -> dict:
    cols = [c for c in df.columns if df[c].isna().any()]
    if not cols:
        return {
            "summary": "No missing values — dataset already has 0% null cells.",
            "actions": [],
        }
    null_cells = int(df.isna().sum().sum())
    return {
        "summary": (
            f"Fill all {null_cells:,} missing cells across {len(cols)} column(s) "
            f"(numeric → median, text → mode, then safe defaults) to reach 0% missing."
        ),
        "actions": [
            {
                "id": "fill_all_nulls",
                "type": "fill_all_nulls",
                "reason": "Remove every remaining null in one step",
            }
        ],
    }


def apply_cleaning_actions(df: pd.DataFrame, actions: list[dict]) -> pd.DataFrame:
    """Apply approved cleaning actions deterministically (no LLM code execution)."""
    out = df.copy()

    for action in actions:
        err = validate_action(out, action)
        if err:
            raise ValueError(err)

        action_type = action["type"]

        if action_type == "skip":
            continue

        if action_type == "drop_column":
            out = out.drop(columns=[action["column"]])

        elif action_type == "fill_null":
            col = action["column"]
            out[col] = _fill_column(
                out[col],
                action.get("strategy", "median"),
                action.get("value"),
            )

        elif action_type == "fill_all_nulls":
            out = _fill_all_nulls(
                out,
                numeric_strategy=action.get("numeric_strategy", "median"),
                categorical_strategy=action.get("categorical_strategy", "mode"),
                categorical_fallback=action.get("categorical_fallback", "Unknown"),
                numeric_fallback=float(action.get("numeric_fallback", 0)),
            )

        elif action_type == "drop_duplicate_rows":
            out = out.drop_duplicates()

        elif action_type == "drop_rows_all_null":
            cols = action.get("columns")
            if cols:
                out = out.dropna(subset=cols, how="all")
            else:
                out = out.dropna(how="all")

        elif action_type == "drop_outlier_rows":
            col = action["column"]
            mask = outlier_mask(
                out[col],
                method=action.get("method", "iqr"),
                factor=float(action.get("factor", 1.5)),
                threshold=float(action.get("threshold", 3.0)),
            )
            out = out.loc[~mask.fillna(False)]

        elif action_type == "cap_outliers":
            col = action["column"]
            lo = float(action.get("lower_percentile", 1)) / 100
            hi = float(action.get("upper_percentile", 99)) / 100
            lower = out[col].quantile(lo)
            upper = out[col].quantile(hi)
            out[col] = out[col].clip(lower=lower, upper=upper)

    return out


def preview_stats(df: pd.DataFrame) -> dict:
    null_pct = df.isna().sum().sum() / max(df.size, 1) * 100
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_pct": round(null_pct, 1),
    }


if __name__ == "__main__":
    sample = pd.DataFrame({
        "id": [1, 1, 2, 3],
        "name": ["a", "a", None, "c"],
        "score": [10.0, 10.0, None, 30.0],
        "empty_col": [None, None, None, None],
    })
    schema = build_schema_string(sample)
    print("Fallback proposal:")
    print(json.dumps(_fallback_proposal(sample), indent=2))
    proposal = _fallback_proposal(sample)
    cleaned = apply_cleaning_actions(sample, proposal["actions"])
    print("\nAfter apply:", preview_stats(cleaned))
