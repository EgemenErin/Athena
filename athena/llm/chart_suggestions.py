import json
import re
from typing import Any, Literal

import numpy as np
import ollama
import pandas as pd

from athena.config import MODEL
from athena.llm.personas import BUSINESS_ANALYST_RULES
from athena.llm.chart_transforms import (
    infer_transform,
    is_multivalue_column,
    multivalue_columns,
)
from athena.llm.schema import (
    build_column_index,
    categorical_columns,
    comparable_numeric_columns,
    looks_numeric_string,
    numeric_columns,
)

ChartType = Literal["bar", "histogram", "scatter", "line", "pie"]

CHART_TYPES: frozenset[str] = frozenset({"bar", "histogram", "scatter", "line", "pie"})
MAX_BAR_CATEGORIES = 25
MAX_PIE_SLICES = 10
MIN_HISTOGRAM_UNIQUE = 5
MAX_HISTOGRAM_UNIQUE_RATIO = 0.5
MAX_CODE_CARDINALITY = 80

_ID_NAME = re.compile(
    r"(^id$|_id$|^id_|response.?id|uuid|guid|(^|_)index($|_)|(^|_)key($|_)|"
    r"seq|serial|row.?num|record.?num|participant.?id|user.?id|survey.?id)",
    re.IGNORECASE,
)

_CODE_DIMENSION_NAME = re.compile(
    r"(district|area|ward|beat|fbi|zip|postal|precinct|tract|block|borough|"
    r"community|neighborhood|grid|sector|offense|crime|location|"
    r"_code$|^code$|_number$|number$|beat)",
    re.IGNORECASE,
)

_METRIC_NAME = re.compile(
    r"(salary|comp|pay|wage|income|revenue|price|cost|amount|score|rating|sat|"
    r"age|exp|percent|rate|hours|size|weight|height|distance|duration|budget|"
    r"profit|loss|margin|conversion|arrest|count|total|population|cases|"
    r"incident|victim|injur|damage|fine|fee|quantity|value|length|width|"
    r"yearscode|workexp|yearsexp)",
    re.IGNORECASE,
)

_DIMENSION_NAME = re.compile(
    r"(country|region|state|city|gender|sex|role|title|job|department|dept|"
    r"team|category|type|status|level|grade|industry|sector|company|employ|"
    r"education|degree|language|platform|channel|source|segment|group|"
    r"district|area|ward|beat|offense|crime|primary|description)",
    re.IGNORECASE,
)


def _is_likely_identifier(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return True
    if _ID_NAME.search(col.replace(" ", "")):
        return True

    series = df[col].dropna()
    n = len(df)
    if n == 0 or len(series) == 0:
        return False

    if not pd.api.types.is_numeric_dtype(df[col]):
        cl = col.lower()
        return cl in ("id", "index", "key") or cl.endswith("_id") or cl.endswith(" id")

    nunique = series.nunique()
    if n >= 20 and nunique >= max(n * 0.95, n - 3):
        return True

    if nunique == n and n >= 20:
        sorted_vals = np.sort(series.to_numpy())
        diffs = np.diff(sorted_vals)
        if len(diffs) > 0 and np.median(diffs) > 0:
            if np.mean(diffs == 1) >= 0.85 or np.mean(np.abs(diffs - np.median(diffs)) < 1e-9) >= 0.85:
                return True

    return False


def _is_integer_like(series: pd.Series) -> bool:
    sample = series.dropna().head(500)
    if len(sample) == 0:
        return False
    if pd.api.types.is_integer_dtype(sample.dtype):
        return True
    numeric = pd.to_numeric(sample, errors="coerce").dropna()
    if len(numeric) == 0:
        return False
    return bool(np.allclose(numeric, np.round(numeric)))


def _is_calendar_year_column(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if not re.search(r"\byear\b", col, re.IGNORECASE) and col.upper() != "YEAR":
        return False
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(vals) == 0:
        return False
    mn, mx = float(vals.min()), float(vals.max())
    return 1900 <= mn <= 2100 and 1900 <= mx <= 2100 and vals.nunique() <= 80


def _is_coded_numeric_dimension(df: pd.DataFrame, col: str) -> bool:
    """Numeric columns that are labels (district #, FBI code), not measurements."""
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if _is_likely_identifier(df, col):
        return True
    if _metric_name_score(col) >= 2:
        return False
    if _CODE_DIMENSION_NAME.search(col.replace(" ", "")):
        return True
    if _is_calendar_year_column(df, col):
        return True

    series = df[col].dropna()
    nunique = series.nunique()
    if nunique < 2 or nunique > MAX_CODE_CARDINALITY:
        return False

    if not _is_integer_like(series):
        return False

    vals = pd.to_numeric(series, errors="coerce").dropna()
    span = float(vals.max() - vals.min())
    if span <= 0:
        return True
    density = nunique / max(span, 1.0)
    if nunique <= 35 and density >= 0.25:
        return True
    if nunique <= 15:
        return True
    return False


def _is_continuous_metric(df: pd.DataFrame, col: str) -> bool:
    """Numeric columns safe for mean, scatter, or distribution (not area codes / years)."""
    if col not in df.columns:
        return False
    if is_multivalue_column(df, col):
        return False
    if looks_numeric_string(df[col]):
        return not _is_likely_identifier(df, col)
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    if _is_likely_identifier(df, col) or _is_coded_numeric_dimension(df, col):
        return False

    series = df[col].dropna()
    nunique = series.nunique()
    if nunique < 2:
        return False

    if _metric_name_score(col) >= 2:
        return True

    if pd.api.types.is_float_dtype(df[col]) and nunique >= 8:
        return True

    if nunique < 20:
        return False

    vals = pd.to_numeric(series, errors="coerce").dropna()
    span = float(vals.max() - vals.min())
    if span <= 0:
        return False
    return (nunique / span) < 0.12


def _has_non_uniform_distribution(df: pd.DataFrame, col: str, bins: int = 20) -> bool:
    series = df[col].dropna()
    if len(series) < MIN_HISTOGRAM_UNIQUE:
        return False
    try:
        counts, _ = np.histogram(series.astype(float), bins=min(bins, max(5, len(series) // 50)))
    except (TypeError, ValueError):
        return False
    counts = counts[counts > 0]
    if len(counts) < 3:
        return False
    mean_c = counts.mean()
    if mean_c <= 0:
        return False
    return float(counts.max() / mean_c) >= 2.0


def _metric_name_score(col: str) -> int:
    if re.fullmatch(r"year", col, re.IGNORECASE):
        return 0
    return 2 if _METRIC_NAME.search(col) else 0


def _dimension_name_score(col: str) -> int:
    return 2 if _DIMENSION_NAME.search(col) else 0


def _continuous_metric_columns(df: pd.DataFrame, max_cols: int = 12) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in comparable_numeric_columns(df):
        if not _is_continuous_metric(df, col):
            continue
        score = float(_metric_name_score(col))
        if _has_non_uniform_distribution(df, col):
            score += 1.0
        scored.append((score, col))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in scored[:max_cols]]


def _histogram_suitable_columns(df: pd.DataFrame, max_cols: int = 4) -> list[tuple[str, str | None]]:
    """Return (column, transform_x) pairs for histograms."""
    scored: list[tuple[float, str, str | None]] = []
    n = len(df)
    for col in df.columns:
        if _is_likely_identifier(df, col) or is_multivalue_column(df, col):
            continue
        transform: str | None = None
        if looks_numeric_string(df[col]):
            transform = "coerce"
            series = pd.to_numeric(df[col].astype(str), errors="coerce").dropna()
        elif _is_continuous_metric(df, col):
            series = df[col].dropna()
        else:
            continue
        nunique = series.nunique()
        if nunique < MIN_HISTOGRAM_UNIQUE:
            continue
        if nunique > max(80, int(n * MAX_HISTOGRAM_UNIQUE_RATIO)):
            transform = "bin"
        elif nunique > 25:
            transform = "bin"
        if transform != "bin" and not _has_non_uniform_distribution(df, col):
            continue
        scored.append((float(_metric_name_score(col)) + 1.0, col, transform))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(c, t) for _, c, t in scored[:max_cols]]


def _insightful_categorical_columns(df: pd.DataFrame, max_cols: int = 10) -> list[str]:
    scored: list[tuple[float, str]] = []
    n = len(df)
    for col in categorical_columns(df):
        nunique = df[col].nunique(dropna=True)
        if nunique < 2 or (n > 0 and nunique > n * 0.95):
            continue
        if _is_likely_identifier(df, col):
            continue
        score = float(_dimension_name_score(col))
        if 3 <= nunique <= MAX_BAR_CATEGORIES:
            score += 2.0
        elif 2 <= nunique <= MAX_PIE_SLICES:
            score += 1.0
        scored.append((score, col))
    scored.sort(key=lambda x: (-x[0], -min(df[x[1]].nunique(dropna=True), 50), x[1]))
    return [c for _, c in scored[:max_cols]]


def _breakdown_columns(df: pd.DataFrame, max_cols: int = 12) -> list[str]:
    """Group-by axes: real categories plus coded numeric dimensions (district, year, etc.)."""
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()

    for col in _insightful_categorical_columns(df, max_cols=max_cols):
        if col in seen:
            continue
        seen.add(col)
        nunique = df[col].nunique(dropna=True)
        score = float(_dimension_name_score(col))
        if 2 <= nunique <= MAX_BAR_CATEGORIES:
            scored.append((score + 2.0, col))

    for col in numeric_columns(df):
        if col in seen:
            continue
        if not (_is_coded_numeric_dimension(df, col) or _is_calendar_year_column(df, col)):
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique < 2 or nunique > MAX_BAR_CATEGORIES:
            continue
        score = float(_dimension_name_score(col)) + 1.5
        if _is_calendar_year_column(df, col):
            score += 1.0
        scored.append((score, col))
        seen.add(col)

    scored.sort(key=lambda x: (-x[0], x[1]))
    breakdowns = [c for _, c in scored[:max_cols]]
    for col in multivalue_columns(df, max_cols=6):
        if col not in breakdowns:
            breakdowns.append(col)
    return breakdowns[:max_cols]


def _meaningful_categorical_columns(df: pd.DataFrame, max_cols: int = 8) -> list[str]:
    return _insightful_categorical_columns(df, max_cols=max_cols)


def _is_datetime_column(df: pd.DataFrame, col: str) -> bool:
    if _is_likely_identifier(df, col) or _is_coded_numeric_dimension(df, col):
        return False
    if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
        return True
    sample = df[col].dropna().head(50)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.8
    except Exception:
        return False


def _spec_key(spec: dict[str, Any]) -> tuple:
    return (
        spec.get("chart_type"),
        spec.get("x"),
        spec.get("y"),
        spec.get("aggregation"),
        spec.get("transform_x"),
        spec.get("transform_y"),
    )


def _fallback_rationale(spec: dict[str, Any]) -> str:
    """Plain-language description when the LLM omits rationale."""
    chart_type = spec.get("chart_type", "chart")
    x = spec.get("x") or "the selected column"
    y = spec.get("y")
    hx, hy = _humanize(str(x)), _humanize(str(y)) if y else ""

    if chart_type == "histogram":
        return f"Shows how values are distributed across {hx}."
    if chart_type == "scatter" and y:
        return f"Plots {hy} against {hx} to reveal correlation or outliers."
    if chart_type == "pie":
        return f"Shows the share of rows in each {hx} category."
    if chart_type == "line":
        if _is_count_bar(spec):
            return f"Tracks how many records appear across {hx}."
        return f"Shows how {hy} changes across {hx}."
    if chart_type == "bar":
        if _is_count_bar(spec):
            return f"Counts rows per {hx} category."
        if y:
            return f"Compares average {hy} across each {hx} group."
        return f"Bar chart grouped by {hx}."
    return f"Visualization using {hx}" + (f" and {hy}" if y else "") + "."


def chart_description(spec: dict[str, Any]) -> str:
    """User-facing description for a chart spec (rationale or generated fallback)."""
    text = str(spec.get("rationale") or spec.get("description") or "").strip()
    if text:
        return text
    return _fallback_rationale(spec)


def _normalize_spec(raw: dict[str, Any]) -> dict[str, Any] | None:
    chart_type = str(raw.get("chart_type", "")).strip().lower()
    if chart_type not in CHART_TYPES:
        return None
    title = str(raw.get("title", "")).strip()
    if not title:
        return None
    x = raw.get("x")
    y = raw.get("y")
    color = raw.get("color")
    agg = raw.get("aggregation")
    tx = raw.get("transform_x")
    ty = raw.get("transform_y")
    x_str = str(x).strip() if x is not None and str(x).strip() else None
    y_str = str(y).strip() if y is not None and str(y).strip() else None
    agg_str = str(agg).strip().lower() if agg else None
    tx_str = str(tx).strip().lower() if tx else None
    ty_str = str(ty).strip().lower() if ty else None
    rationale = str(raw.get("rationale") or raw.get("description") or "").strip()
    if not rationale:
        rationale = _fallback_rationale({
            "chart_type": chart_type,
            "x": x_str,
            "y": y_str,
            "aggregation": agg_str,
        })
    return {
        "chart_type": chart_type,
        "x": x_str,
        "y": y_str,
        "color": str(color).strip() if color is not None and str(color).strip() else None,
        "aggregation": agg_str,
        "transform_x": tx_str,
        "transform_y": ty_str,
        "title": title[:120],
        "rationale": rationale[:240],
    }


def _is_count_bar(spec: dict[str, Any]) -> bool:
    return spec.get("aggregation") == "count" or spec.get("y") == "count"


def validate_chart_spec(df: pd.DataFrame, spec: dict[str, Any]) -> bool:
    normalized = _normalize_spec(spec)
    if not normalized:
        return False

    chart_type = normalized["chart_type"]
    x, y = normalized["x"], normalized["y"]
    breakdowns = set(_breakdown_columns(df, max_cols=200))

    def _col_ok(name: str | None) -> bool:
        return name is not None and name in df.columns

    if chart_type == "histogram":
        if not _col_ok(x):
            return False
        suitable = {c for c, _ in _histogram_suitable_columns(df, max_cols=100)}
        if x not in suitable and not _is_continuous_metric(df, x):
            return False
        return _validate_renderable(df, normalized)

    if chart_type == "scatter":
        if not _col_ok(x) or not _col_ok(y) or x == y:
            return False
        if not (_is_continuous_metric(df, x) and _is_continuous_metric(df, y)):
            return False
        return _validate_renderable(df, normalized)

    if chart_type == "line":
        if not _col_ok(x):
            return False
        if _is_count_bar(normalized):
            return x in breakdowns and _is_calendar_year_column(df, x)
        if not _col_ok(y) or not _is_continuous_metric(df, y):
            return False
        return _is_calendar_year_column(df, x) or _is_datetime_column(df, x)

    if chart_type == "pie":
        if not _col_ok(x) or _is_likely_identifier(df, x):
            return False
        nunique = df[x].nunique(dropna=True)
        return 2 <= nunique <= MAX_PIE_SLICES

    if chart_type == "bar":
        if not _col_ok(x):
            return False
        if is_multivalue_column(df, x):
            if normalized.get("transform_x") not in (None, "explode"):
                tx = infer_transform(df, x, chart_type="bar", role="x")
                if tx != "explode":
                    return False
        elif x not in breakdowns and not _is_user_breakdown_axis(df, x):
            return False
        if _is_count_bar(normalized):
            return _validate_renderable(df, normalized)
        if not _col_ok(y) or not _is_continuous_metric(df, y):
            return False
        return _validate_renderable(df, normalized)

    return False


def _validate_renderable(df: pd.DataFrame, spec: dict[str, Any]) -> bool:
    from athena.ui.charts import prepare_chart_data

    filled = dict(spec)
    if not filled.get("transform_x") and filled.get("x"):
        filled["transform_x"] = infer_transform(
            df, filled["x"], chart_type=filled["chart_type"], role="x"
        )
    if not filled.get("transform_y") and filled.get("y") and filled.get("y") != "count":
        filled["transform_y"] = infer_transform(
            df, filled["y"], chart_type=filled["chart_type"], role="y"
        )
    plot_df = prepare_chart_data(df, filled)
    return plot_df is not None and not plot_df.empty


def _make_spec(
    chart_type: ChartType,
    *,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    aggregation: str | None = None,
    transform_x: str | None = None,
    transform_y: str | None = None,
    title: str,
    rationale: str,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "chart_type": chart_type,
        "x": x,
        "y": y,
        "color": color,
        "title": title,
        "rationale": rationale,
    }
    if aggregation:
        spec["aggregation"] = aggregation
    if transform_x:
        spec["transform_x"] = transform_x
    if transform_y:
        spec["transform_y"] = transform_y
    return spec


def _humanize(col: str) -> str:
    return col.replace("_", " ")


def _chart_variant(spec: dict[str, Any]) -> str:
    chart_type = spec.get("chart_type", "")
    if chart_type == "line":
        return "line_count" if _is_count_bar(spec) else "line_mean"
    if chart_type == "bar":
        return "bar_count" if _is_count_bar(spec) else "bar_mean"
    return chart_type


def _candidate_score(df: pd.DataFrame, spec: dict[str, Any], *, llm_boost: float = 0) -> float:
    score = llm_boost
    x = spec.get("x")
    y = spec.get("y")
    if x and x in df.columns:
        score += float(_dimension_name_score(x)) * 2.0
        nunique = df[x].nunique(dropna=True)
        if 3 <= nunique <= 15:
            score += 2.0
        elif 2 <= nunique <= MAX_BAR_CATEGORIES:
            score += 1.0
    if y and y in df.columns and y != "count":
        score += float(_metric_name_score(y))
    variant = _chart_variant(spec)
    if variant == "histogram":
        score += 1.5
    elif variant == "scatter":
        score += 1.5
    elif variant == "line_count":
        score += 2.0
    return score


def _select_diverse_specs(
    candidates: list[dict[str, Any]],
    df: pd.DataFrame,
    n: int,
) -> list[dict[str, Any]]:
    """Pick up to n charts: different chart kinds and different group-by columns when possible."""
    scored: list[tuple[float, dict[str, Any]]] = []
    seen_keys: set[tuple] = set()

    for spec in candidates:
        normalized = _normalize_spec(spec)
        if not normalized or not validate_chart_spec(df, normalized):
            continue
        key = _spec_key(normalized)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        boost = 5.0 if spec.get("_llm") else 0.0
        scored.append((_candidate_score(df, normalized, llm_boost=boost), normalized))

    scored.sort(key=lambda item: (-item[0], item[1].get("title", "")))
    ranked = [spec for _, spec in scored]

    out: list[dict[str, Any]] = []
    used_x: set[str] = set()
    used_hist_metric: set[str] = set()
    used_scatter_pair: set[tuple[str, str]] = set()

    variant_order = [
        "line_count",
        "bar_count",
        "bar_mean",
        "pie",
        "histogram",
        "scatter",
        "line_mean",
    ]

    def _take(spec: dict[str, Any]) -> None:
        out.append(spec)
        x = spec.get("x")
        if not x:
            return
        chart_type = spec.get("chart_type")
        if chart_type in ("bar", "pie", "line"):
            used_x.add(x)
        elif chart_type == "histogram":
            used_hist_metric.add(x)
        elif chart_type == "scatter" and spec.get("y"):
            used_scatter_pair.add(tuple(sorted((x, spec["y"]))))

    def _blocked(spec: dict[str, Any]) -> bool:
        chart_type = spec.get("chart_type")
        x, y = spec.get("x"), spec.get("y")
        if chart_type in ("bar", "pie", "line") and x and x in used_x:
            return True
        if chart_type == "histogram" and x and x in used_hist_metric:
            return True
        if chart_type == "scatter" and x and y:
            pair = tuple(sorted((x, y)))
            if pair in used_scatter_pair:
                return True
        return False

    for target_variant in variant_order:
        if len(out) >= n:
            break
        for spec in ranked:
            if _chart_variant(spec) != target_variant or _blocked(spec):
                continue
            _take(spec)
            break

    for spec in ranked:
        if len(out) >= n:
            break
        if spec in out or _blocked(spec):
            continue
        _take(spec)

    return out[:n]


def _collect_fallback_candidates(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Build all valid chart ideas; diversity selection picks the final six."""
    metrics = _continuous_metric_columns(df)
    breakdowns = _breakdown_columns(df)
    hist_cols = _histogram_suitable_columns(df)
    candidates: list[dict[str, Any]] = []

    for col in multivalue_columns(df, max_cols=6):
        candidates.append(
            _make_spec(
                "bar",
                x=col,
                y="count",
                aggregation="count",
                transform_x="explode",
                title=f"Most common values in {_humanize(col)}",
                rationale=(
                    f"Splits multi-value `{col}` cells (e.g. semicolon lists) "
                    f"into separate rows, then counts each item."
                ),
            )
        )

    for dim in breakdowns:
        nunique = df[dim].nunique(dropna=True)
        if not (2 <= nunique <= MAX_BAR_CATEGORIES):
            continue
        candidates.append(
            _make_spec(
                "bar",
                x=dim,
                y="count",
                aggregation="count",
                title=f"How many records per {_humanize(dim)}",
                rationale=f"Volume of rows across each `{dim}` value ({nunique} groups).",
            )
        )

    for dim in breakdowns:
        if _is_calendar_year_column(df, dim):
            candidates.append(
                _make_spec(
                    "line",
                    x=dim,
                    y="count",
                    aggregation="count",
                    title=f"Records over {_humanize(dim)}",
                    rationale=f"How row volume changes year by year in `{dim}`.",
                )
            )

    for dim in breakdowns:
        for metric in metrics[:8]:
            nunique = df[dim].nunique(dropna=True)
            if 2 <= nunique <= MAX_BAR_CATEGORIES:
                ty = "coerce" if looks_numeric_string(df[metric]) else None
                candidates.append(
                    _make_spec(
                        "bar",
                        x=dim,
                        y=metric,
                        transform_y=ty,
                        title=f"Average {_humanize(metric)} by {_humanize(dim)}",
                        rationale=(
                            f"Typical `{metric}` in each `{dim}` group — "
                            f"compare levels across {nunique} categories."
                        ),
                    )
                )

    for dim in breakdowns:
        nunique = df[dim].nunique(dropna=True)
        if 2 <= nunique <= MAX_PIE_SLICES and not _is_calendar_year_column(df, dim):
            candidates.append(
                _make_spec(
                    "pie",
                    x=dim,
                    title=f"Share of rows by {_humanize(dim)}",
                    rationale=f"Mix of records across `{dim}` ({nunique} groups).",
                )
            )

    for col, tx in hist_cols:
        label = "binned ranges" if tx == "bin" else "values"
        if tx == "bin":
            suffix = ", grouped into readable ranges."
        elif tx == "coerce":
            suffix = ", parsed from text into numbers."
        else:
            suffix = "."
        candidates.append(
            _make_spec(
                "histogram",
                x=col,
                transform_x=tx,
                title=f"Distribution of {_humanize(col)} ({label})",
                rationale=f"Spread of `{col}`{suffix}",
            )
        )

    if len(metrics) >= 2:
        for i, x_col in enumerate(metrics[:5]):
            for y_col in metrics[i + 1 : i + 3]:
                candidates.append(
                    _make_spec(
                        "scatter",
                        x=x_col,
                        y=y_col,
                        title=f"{_humanize(y_col)} vs {_humanize(x_col)}",
                        rationale=(
                            f"Relationship between two measured fields: "
                            f"`{x_col}` and `{y_col}`."
                        ),
                    )
                )

    for col in df.columns:
        if not _is_datetime_column(df, col):
            continue
        for metric in metrics[:4]:
            if col == metric:
                continue
            candidates.append(
                _make_spec(
                    "line",
                    x=col,
                    y=metric,
                    title=f"{_humanize(metric)} over time",
                    rationale=f"How `{metric}` changes across `{col}`.",
                )
            )

    return candidates


def _matches_column_selection(spec: dict[str, Any], x: str, y: str | None) -> bool:
    if spec.get("x") != x:
        return False
    if y is None:
        return True
    spec_y = spec.get("y")
    if spec_y == y:
        return True
    return y == "count" and _is_count_bar(spec)


def _is_user_breakdown_axis(df: pd.DataFrame, col: str) -> bool:
    """Looser group-by check for explicit user column picks (small datasets allowed)."""
    if col not in df.columns or _is_likely_identifier(df, col):
        return False
    nunique = df[col].nunique(dropna=True)
    return 2 <= nunique <= MAX_BAR_CATEGORIES


def _synthesize_candidates_for_columns(
    df: pd.DataFrame,
    x: str,
    y: str | None,
) -> list[dict[str, Any]]:
    """Build chart specs when no fallback candidate matches the user's column pick."""
    if x not in df.columns:
        return []

    candidates: list[dict[str, Any]] = []
    metrics = _continuous_metric_columns(df)
    nunique = df[x].nunique(dropna=True)
    user_breakdown = _is_user_breakdown_axis(df, x)

    if is_multivalue_column(df, x):
        candidates.append(
            _make_spec(
                "bar",
                x=x,
                y="count",
                aggregation="count",
                transform_x="explode",
                title=f"Most common values in {_humanize(x)}",
                rationale=f"Counts each value in multi-value `{x}` cells.",
            )
        )

    if user_breakdown:
        candidates.append(
            _make_spec(
                "bar",
                x=x,
                y="count",
                aggregation="count",
                title=f"How many records per {_humanize(x)}",
                rationale=f"Row volume across each `{x}` value.",
            )
        )
        if 2 <= nunique <= MAX_PIE_SLICES and not _is_calendar_year_column(df, x):
            candidates.append(
                _make_spec(
                    "pie",
                    x=x,
                    title=f"Share of rows by {_humanize(x)}",
                    rationale=f"Mix of records across `{x}`.",
                )
            )
        if _is_calendar_year_column(df, x):
            candidates.append(
                _make_spec(
                    "line",
                    x=x,
                    y="count",
                    aggregation="count",
                    title=f"Records over {_humanize(x)}",
                    rationale=f"How row volume changes across `{x}`.",
                )
            )

    if y and y in df.columns and y != "count" and _is_continuous_metric(df, y):
        if user_breakdown:
            ty = "coerce" if looks_numeric_string(df[y]) else None
            candidates.append(
                _make_spec(
                    "bar",
                    x=x,
                    y=y,
                    transform_y=ty,
                    title=f"Average {_humanize(y)} by {_humanize(x)}",
                    rationale=f"Typical `{y}` in each `{x}` group.",
                )
            )
        if (
            _is_continuous_metric(df, x)
            and _is_continuous_metric(df, y)
            and x != y
        ):
            candidates.append(
                _make_spec(
                    "scatter",
                    x=x,
                    y=y,
                    title=f"{_humanize(y)} vs {_humanize(x)}",
                    rationale=f"Relationship between `{x}` and `{y}`.",
                )
            )
        if _is_datetime_column(df, x) and x != y:
            candidates.append(
                _make_spec(
                    "line",
                    x=x,
                    y=y,
                    title=f"{_humanize(y)} over time",
                    rationale=f"How `{y}` changes across `{x}`.",
                )
            )

    for col, tx in _histogram_suitable_columns(df, max_cols=50):
        if col != x:
            continue
        label = "binned ranges" if tx == "bin" else "values"
        candidates.append(
            _make_spec(
                "histogram",
                x=x,
                transform_x=tx,
                title=f"Distribution of {_humanize(x)} ({label})",
                rationale=f"Spread of `{x}`.",
            )
        )
        break

    if y is None and _is_continuous_metric(df, x) and not any(
        c.get("chart_type") == "histogram" and c.get("x") == x for c in candidates
    ):
        candidates.append(
            _make_spec(
                "histogram",
                x=x,
                transform_x="coerce" if looks_numeric_string(df[x]) else None,
                title=f"Distribution of {_humanize(x)}",
                rationale=f"Spread of `{x}` values.",
            )
        )

    return candidates


def suggest_charts_for_columns(
    df: pd.DataFrame,
    x: str,
    y: str | None = None,
    *,
    max_options: int = 5,
) -> list[dict[str, Any]]:
    """Heuristic chart ideas for user-selected columns (no LLM call)."""
    if x not in df.columns:
        return []

    pool = _collect_fallback_candidates(df)
    matched = [s for s in pool if _matches_column_selection(s, x, y)]
    matched.extend(_synthesize_candidates_for_columns(df, x, y))

    seen: set[tuple] = set()
    scored: list[tuple[float, dict[str, Any]]] = []
    for spec in matched:
        normalized = _normalize_spec(spec)
        if not normalized or not validate_chart_spec(df, normalized):
            continue
        key = _spec_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        scored.append((_candidate_score(df, normalized), normalized))

    scored.sort(key=lambda item: (-item[0], item[1].get("title", "")))
    return [spec for _, spec in scored[:max_options]]


def _fallback_chart_suggestions(df: pd.DataFrame, n: int = 6) -> list[dict[str, Any]]:
    return _select_diverse_specs(_collect_fallback_candidates(df), df, n)


def _parse_llm_specs(raw: str, n: int) -> list[dict[str, Any]]:
    text = raw.strip()
    if not text:
        return []

    specs: list[dict[str, Any]] = []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    normalized = _normalize_spec(item)
                    if normalized:
                        specs.append(normalized)
            return specs[: n * 2]
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\d+[\.\):]\s*", "", line)
        if line.startswith("{") and line.endswith("}"):
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    normalized = _normalize_spec(item)
                    if normalized:
                        specs.append(normalized)
            except json.JSONDecodeError:
                continue

    return specs[: n * 2]


def _merge_specs(
    primary: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    df: pd.DataFrame,
    n: int,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for spec in primary:
        marked = dict(spec)
        marked["_llm"] = True
        pool.append(marked)
    pool.extend(fallback)
    return _select_diverse_specs(pool, df, n)


def generate_chart_suggestions(
    df: pd.DataFrame,
    schema: str,
    n: int = 6,
) -> list[dict[str, Any]]:
    """Dataset-specific chart suggestions via Ollama with rule-based fallback."""
    metrics = _continuous_metric_columns(df)[:15]
    breakdowns = _breakdown_columns(df, max_cols=12)
    column_index = build_column_index(df)
    schema_excerpt = schema if len(schema) <= 4000 else schema[:4000] + "\n... (truncated)"

    prompt = (
        f"{BUSINESS_ANALYST_RULES}\n\n"
        f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns.\n"
        f"Measured metrics (for averages / scatter): "
        f"{', '.join(f'`{c}`' for c in metrics) or 'none'}\n"
        f"Group-by columns (categories / codes / year): "
        f"{', '.join(f'`{c}`' for c in breakdowns) or 'none'}\n\n"
        f"{column_index}\n\n"
        f"Schema detail:\n{schema_excerpt}\n\n"
        f"Suggest exactly {n} charts as a JSON array. Each object:\n"
        '- "chart_type": bar, histogram, scatter, line, or pie\n'
        '- "x", "y": exact column names (y may be null for pie)\n'
        '- "aggregation": optional — use "count" for bar/line showing row volume by group\n'
        '- "transform_x" / "transform_y": optional — "coerce" (parse numeric text), '
        '"bin" (bucket into ranges), "explode" (split list-like cells on ; or |)\n'
        '- "title": short chart headline\n'
        '- "rationale": 1–2 sentences — what this chart shows and what decision it supports '
        '(required; also accepted as "description")\n\n'
        "Transforms: use explode for multi-value text columns; coerce for numeric strings; "
        "bin for high-cardinality metrics.\n\n"
        "STRICT rules:\n"
        "- NEVER chart ID/ResponseId/uuid columns.\n"
        "- NEVER scatter geographic/admin codes (District, Area Number, FBI Code, Zip) "
        "against each other or against Year — those are labels, not measurements.\n"
        "- District / Community Area / FBI Code / Year are GROUP-BY axes only "
        '(bar with aggregation "count", or average of a real metric).\n'
        "- Scatter ONLY between two columns from the measured metrics list.\n"
        "- Use 6 DIFFERENT primary groupings (x): never repeat the same x column twice.\n"
        "- Vary chart_type across the six (mix bar, line, pie, histogram, scatter when valid).\n"
        "- Prefer: (1) count by one district/year/type, (2) average metric by a different group, "
        "(3) trend over year, (4) pie for another category, (5) histogram of a metric, "
        "(6) scatter of two metrics.\n"
        "JSON array only:\n"
    )

    pool = _collect_fallback_candidates(df)
    llm_specs: list[dict[str, Any]] = []
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.35, "num_predict": 900},
        )
        llm_specs = _parse_llm_specs(response["message"]["content"].strip(), n)
    except Exception:
        pass

    return _merge_specs(llm_specs, pool, df, n)
