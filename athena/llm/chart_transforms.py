"""Prepare non-numeric or messy columns for dashboard charts (coerce, bin, explode)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from athena.llm.schema import looks_numeric_string

MULTI_VALUE_SEPARATORS = (";", "|", ",")
DEFAULT_BIN_COUNT = 10
MAX_EXPLODE_CATEGORIES = 20


def detect_multivalue_separator(series: pd.Series) -> str | None:
    sample = series.dropna().astype(str).head(200)
    if len(sample) == 0:
        return None
    best_sep: str | None = None
    best_hits = 0.0
    for sep in MULTI_VALUE_SEPARATORS:
        hits = sample.str.contains(re.escape(sep), regex=False).mean()
        if hits > best_hits:
            best_hits = hits
            best_sep = sep
    return best_sep if best_hits >= 0.25 else None


def is_multivalue_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    if pd.api.types.is_numeric_dtype(df[col].dtype):
        return False
    return detect_multivalue_separator(df[col]) is not None


def multivalue_columns(df: pd.DataFrame, max_cols: int = 8) -> list[str]:
    out: list[str] = []
    for col in df.columns:
        if is_multivalue_column(df, col):
            out.append(col)
        if len(out) >= max_cols:
            break
    return out


def coerce_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col].astype(str), errors="coerce")
    return out


def bin_column(
    df: pd.DataFrame,
    col: str,
    *,
    n_bins: int = DEFAULT_BIN_COUNT,
) -> tuple[pd.DataFrame, str]:
    """Bucket a numeric column into readable ranges for bar/histogram axes."""
    out = df.copy()
    values = pd.to_numeric(out[col], errors="coerce")
    valid = values.dropna()
    if len(valid) < 2:
        binned_col = col
        return out, binned_col

    n_bins = min(n_bins, max(2, valid.nunique()))
    try:
        buckets = pd.qcut(valid, q=n_bins, duplicates="drop")
    except ValueError:
        buckets = pd.cut(valid, bins=n_bins, duplicates="drop")

    binned_col = f"{col} (binned)"
    out[binned_col] = pd.Series(index=out.index, dtype=object)
    out.loc[valid.index, binned_col] = buckets.astype(str).values
    out = out.dropna(subset=[binned_col])
    return out, binned_col


def _clean_split_parts(parts) -> list[str]:
    """Normalize str.split output; NaN cells are not lists."""
    if parts is None:
        return []
    if isinstance(parts, float) and pd.isna(parts):
        return []
    if isinstance(parts, str):
        text = parts.strip()
        return [] if not text or text.lower() == "nan" else [text]
    try:
        return [
            str(p).strip()
            for p in parts
            if p is not None
            and str(p).strip()
            and str(p).strip().lower() != "nan"
        ]
    except TypeError:
        return []


def explode_column(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, str]:
    """Split semicolon/pipe/comma lists into one row per value."""
    sep = detect_multivalue_separator(df[col]) or ";"
    out = df.copy()
    out[col] = (
        out[col]
        .astype(str)
        .str.split(sep)
        .apply(_clean_split_parts)
    )
    out = out.explode(col, ignore_index=True)
    out = out[out[col].notna() & (out[col].astype(str).str.len() > 0)]
    return out, col


def infer_transform(
    df: pd.DataFrame,
    col: str | None,
    *,
    chart_type: str,
    role: str,
) -> str | None:
    if not col or col not in df.columns:
        return None

    if chart_type in ("bar", "pie") and role == "x":
        if is_multivalue_column(df, col):
            return "explode"
        return None

    if chart_type == "histogram" and role == "x":
        if looks_numeric_string(df[col]):
            return "coerce"
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            nunique = df[col].nunique(dropna=True)
            if nunique > 25:
                return "bin"
        return None

    if role == "y" and col != "count":
        if looks_numeric_string(df[col]):
            return "coerce"
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            nunique = df[col].nunique(dropna=True)
            if nunique > 25 and chart_type == "bar":
                return None
        return None

    if chart_type == "scatter" and role in ("x", "y"):
        if looks_numeric_string(df[col]) or not pd.api.types.is_numeric_dtype(df[col].dtype):
            return "coerce"
        return None

    return None


def apply_transforms(
    df: pd.DataFrame,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Return a plotting frame plus resolved x/y column names after transforms."""
    chart_type = spec.get("chart_type")
    x = spec.get("x")
    y = spec.get("y")

    if not chart_type or not x or x not in df.columns:
        return None, None, None

    work = df
    plot_x = x
    plot_y = y

    tx = spec.get("transform_x") or infer_transform(work, x, chart_type=chart_type, role="x")
    ty = None
    if y and y != "count":
        ty = spec.get("transform_y") or infer_transform(
            work, y, chart_type=chart_type, role="y"
        )

    if tx == "explode":
        work, plot_x = explode_column(work, x)
    elif tx == "coerce":
        work = coerce_column(work, x)
    elif tx == "bin":
        work, plot_x = bin_column(work, x)

    if ty == "coerce" and plot_y and plot_y in work.columns:
        work = coerce_column(work, plot_y)
    elif ty == "bin" and plot_y and plot_y in work.columns:
        work, plot_y = bin_column(work, plot_y)

    if chart_type == "histogram":
        if plot_x not in work.columns:
            return None, None, None
        return work[[plot_x]].dropna(), plot_x, None

    if chart_type == "scatter":
        if not plot_y or plot_y not in work.columns:
            return None, None, None
        for col in (plot_x, plot_y):
            if col in work.columns and (
                looks_numeric_string(work[col])
                or infer_transform(work, col, chart_type="scatter", role="x") == "coerce"
            ):
                work = coerce_column(work, col)
        return work[[plot_x, plot_y]].dropna(), plot_x, plot_y

    return work, plot_x, plot_y
