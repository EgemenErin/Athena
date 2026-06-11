from typing import Any

import pandas as pd
import plotly.express as px

from athena.charts.constants import (
    AGGREGATION_LABELS,
    MAX_BAR_CATEGORIES,
    MAX_LINE_POINTS,
    MAX_PIE_SLICES,
)
from athena.llm.chart_transforms import apply_transforms

# Matches the app CSS in athena/ui/styles.py (light paper theme, ink text, accent blue).
# Arial fallback keeps fonts Kaleido-safe for PDF export.
CATEGORICAL_PALETTE = [
    "#2D4FDE", "#0E8A6D", "#C5483D", "#B8860B", "#7C5CD6",
    "#1A7F8E", "#A8533E", "#5B7A2A", "#8A4F9E", "#3D6B9B",
]
SEQUENTIAL_SCALE = ["#C5D0F6", "#2D4FDE"]

PLOTLY_THEME = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#1C1B17",
    font_family="Instrument Sans, Arial, sans-serif",
    colorway=CATEGORICAL_PALETTE,
)


def _apply_plotly_theme(fig, *, tick_angle: int = -35, show_legend: bool = False) -> None:
    fig.update_layout(
        **PLOTLY_THEME,
        xaxis_tickangle=tick_angle,
        showlegend=show_legend,
        margin=dict(t=30, b=60, l=40, r=20),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(gridcolor="#ECE9DF", linecolor="#E5E2D8", zerolinecolor="#E5E2D8")
    fig.update_yaxes(gridcolor="#ECE9DF", linecolor="#E5E2D8", zerolinecolor="#E5E2D8")
    fig.update_traces(marker_line_width=0)


_MONTH_ORDER = {
    name: i
    for i, names in enumerate([
        ("jan", "january"), ("feb", "february"), ("mar", "march"),
        ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
        ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
        ("nov", "november"), ("dec", "december"),
    ])
    for name in names
}

_WEEKDAY_ORDER = {
    name: i
    for i, names in enumerate([
        ("mon", "monday"), ("tue", "tues", "tuesday"),
        ("wed", "wednesday"), ("thu", "thur", "thurs", "thursday"),
        ("fri", "friday"), ("sat", "saturday"), ("sun", "sunday"),
    ])
    for name in names
}


def _period_order(values: pd.Series) -> pd.Series | None:
    """Sort key for month/weekday name axes, or None when not applicable."""
    lowered = values.astype(str).str.strip().str.lower()
    unique = set(lowered.dropna().unique())
    for mapping in (_MONTH_ORDER, _WEEKDAY_ORDER):
        if unique and unique <= set(mapping):
            return lowered.map(mapping)
    return None


def _sort_line_axis(plot_df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    order = _period_order(plot_df[x_col])
    if order is not None:
        return plot_df.assign(_order=order).sort_values("_order").drop(columns="_order")
    return plot_df.sort_values(x_col)


def _truncate_grouped(
    grouped: pd.DataFrame,
    x_col: str,
    value_col: str,
    max_items: int,
    *,
    additive: bool,
) -> pd.DataFrame:
    """
    Cap category counts. Additive aggregations (count/sum) fold the remainder
    into an "Other (N)" bucket; others truncate. Either way a note is attached
    via DataFrame.attrs so the UI can explain what was cut.
    """
    total = len(grouped)
    if total <= max_items:
        return grouped

    if additive:
        top = grouped.head(max_items - 1)
        rest = grouped.iloc[max_items - 1 :]
        other_row = pd.DataFrame({
            x_col: [f"Other ({len(rest)})"],
            value_col: [rest[value_col].sum()],
        })
        out = pd.concat([top, other_row], ignore_index=True)
        note = (
            f"Showing top {max_items - 1} of {total} categories — "
            f"the remaining {len(rest)} are grouped as “Other”."
        )
    else:
        out = grouped.head(max_items).reset_index(drop=True)
        note = f"Showing top {max_items} of {total} categories."

    out.attrs["truncation_note"] = note
    return out


def _flag_to_float(series: pd.Series) -> pd.Series:
    """Boolean / 0-1 flag values as floats for rate calculations."""
    try:
        return series.astype("Float64").astype("float64")
    except (TypeError, ValueError):
        return pd.to_numeric(series, errors="coerce")


def prepare_chart_data(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame | None:
    """Build a plotting-ready frame from a chart suggestion spec."""
    chart_type = spec.get("chart_type")
    x = spec.get("x")
    y = spec.get("y")

    if chart_type in ("histogram", "scatter"):
        plot_df, plot_x, plot_y = apply_transforms(df, spec)
        return plot_df

    work, plot_x, plot_y = apply_transforms(df, spec)
    if work is None or not plot_x or plot_x not in work.columns:
        return None

    if chart_type == "line":
        agg = spec.get("aggregation", "mean")
        if agg == "count" or y == "count":
            plot_df = (
                work.groupby(plot_x, dropna=False)
                .size()
                .reset_index(name="count")
            )
            plot_df = _sort_line_axis(plot_df, plot_x)
            return plot_df.head(MAX_LINE_POINTS)
        if not plot_y or plot_y not in work.columns:
            return None
        plot_df = work[[plot_x, plot_y]].dropna().copy()
        if pd.api.types.is_datetime64_any_dtype(plot_df[plot_x].dtype):
            plot_df = plot_df.sort_values(plot_x)
        elif _period_order(plot_df[plot_x]) is not None:
            plot_df = (
                plot_df.groupby(plot_x, dropna=False)[plot_y]
                .mean()
                .reset_index(name=plot_y)
            )
            plot_df = _sort_line_axis(plot_df, plot_x)
        else:
            try:
                numeric_x = pd.to_numeric(plot_df[plot_x], errors="coerce")
                if numeric_x.notna().mean() >= 0.9:
                    plot_df = plot_df.assign(**{plot_x: numeric_x}).dropna(subset=[plot_x])
                    plot_df = plot_df.sort_values(plot_x)
                else:
                    plot_df[plot_x] = pd.to_datetime(plot_df[plot_x], errors="coerce")
                    plot_df = plot_df.dropna(subset=[plot_x]).sort_values(plot_x)
            except Exception:
                plot_df = plot_df.sort_values(plot_x)
        return plot_df.head(MAX_LINE_POINTS)

    if chart_type == "pie":
        counts = work[plot_x].value_counts(dropna=True)
        if len(counts) < 2:
            return None
        grouped = counts.reset_index(name="count")
        return _truncate_grouped(
            grouped, grouped.columns[0], "count", MAX_PIE_SLICES,
            additive=True,
        )

    if chart_type == "bar":
        agg = spec.get("aggregation") or ("count" if y == "count" else "mean")
        if agg == "count" or y == "count":
            grouped = (
                work.groupby(plot_x, dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .reset_index(drop=True)
            )
            return _truncate_grouped(
                grouped, plot_x, "count", MAX_BAR_CATEGORIES,
                additive=True,
            )
        if not plot_y or plot_y not in work.columns:
            return None
        if agg == "pct_true":
            flags = _flag_to_float(work[plot_y])
            grouped = (
                flags.groupby(work[plot_x], dropna=False)
                .mean()
                .mul(100)
                .reset_index(name=plot_y)
                .sort_values(plot_y, ascending=False)
                .reset_index(drop=True)
            )
            grouped = grouped.dropna(subset=[plot_y])
            if grouped.empty:
                return None
            return _truncate_grouped(
                grouped, plot_x, plot_y, MAX_BAR_CATEGORIES,
                additive=False,
            )
        if agg not in ("mean", "median", "sum", "min", "max"):
            agg = "mean"
        grouped = (
            work.groupby(plot_x, dropna=False)[plot_y]
            .agg(agg)
            .reset_index(name=plot_y)
            .sort_values(plot_y, ascending=False)
            .reset_index(drop=True)
        )
        return _truncate_grouped(
            grouped, plot_x, plot_y, MAX_BAR_CATEGORIES,
            additive=(agg == "sum"),
        )

    return None


def _attach_truncation_note(fig, plot_df: pd.DataFrame) -> None:
    note = plot_df.attrs.get("truncation_note")
    if note:
        fig.update_layout(meta={"truncation_note": note})


def chart_truncation_note(fig) -> str | None:
    """Read the truncation caption stored on a rendered figure, if any."""
    meta = getattr(fig.layout, "meta", None)
    if isinstance(meta, dict):
        return meta.get("truncation_note")
    return None


def _is_count_spec(spec: dict[str, Any]) -> bool:
    return spec.get("aggregation") == "count" or spec.get("y") == "count"


def build_chart_from_spec(df: pd.DataFrame, spec: dict[str, Any]):
    """Render a Plotly figure from a validated chart suggestion spec."""
    plot_df = prepare_chart_data(df, spec)
    if plot_df is None or plot_df.empty:
        return None

    chart_type = spec.get("chart_type")
    x = spec.get("x")
    y = spec.get("y")
    title = spec.get("title") or ""

    if chart_type == "histogram" and len(plot_df.columns) >= 1:
        x_col = plot_df.columns[0]
        fig = px.histogram(plot_df, x=x_col, title=title)
        _apply_plotly_theme(fig, tick_angle=0)
        return fig

    if chart_type == "scatter" and x in plot_df.columns and y in plot_df.columns:
        fig = px.scatter(plot_df, x=x, y=y, title=title)
        _apply_plotly_theme(fig, tick_angle=0)
        return fig

    if chart_type == "line" and len(plot_df.columns) >= 2:
        x_col, y_plot = plot_df.columns[0], plot_df.columns[1]
        fig = px.line(plot_df, x=x_col, y=y_plot, title=title)
        _apply_plotly_theme(fig, tick_angle=-25)
        return fig

    if chart_type == "pie":
        name_col = plot_df.columns[0]
        fig = px.pie(
            plot_df,
            names=name_col,
            values="count",
            title=title,
            color_discrete_sequence=CATEGORICAL_PALETTE,
        )
        _apply_plotly_theme(fig, tick_angle=0, show_legend=True)
        _attach_truncation_note(fig, plot_df)
        return fig

    if chart_type == "bar" and len(plot_df.columns) >= 2:
        x_col, y_col = plot_df.columns[0], plot_df.columns[1]
        agg = spec.get("aggregation")
        if _is_count_spec(spec):
            # Count bars: categories are nominal — use the categorical palette.
            fig = px.bar(
                plot_df,
                x=x_col,
                y=y_col,
                color=x_col,
                title=title,
                color_discrete_sequence=CATEGORICAL_PALETTE,
            )
        else:
            # Value bars (mean/median/sum/…): sequential scale encodes magnitude.
            fig = px.bar(
                plot_df,
                x=x_col,
                y=y_col,
                color=y_col,
                title=title,
                color_continuous_scale=SEQUENTIAL_SCALE,
            )
        if agg == "pct_true":
            fig.update_yaxes(title_text=f"% {y} = True", ticksuffix="%")
        elif agg in AGGREGATION_LABELS and agg not in ("count", None):
            fig.update_yaxes(title_text=f"{AGGREGATION_LABELS[agg]} {y}")
        _apply_plotly_theme(fig)
        _attach_truncation_note(fig, plot_df)
        return fig

    return None


def _is_square_numeric_matrix(df: pd.DataFrame) -> bool:
    rows, cols = df.shape
    if rows != cols or rows == 0 or cols > 6:
        return False
    numeric = df.select_dtypes(include="number")
    return len(numeric.columns) == cols


def _pick_xy_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    if df.shape[1] != 2:
        return None
    c0, c1 = df.columns[0], df.columns[1]
    if pd.api.types.is_numeric_dtype(df[c1]) and not pd.api.types.is_numeric_dtype(df[c0]):
        return c0, c1
    if pd.api.types.is_numeric_dtype(df[c0]) and not pd.api.types.is_numeric_dtype(df[c1]):
        return c1, c0
    if pd.api.types.is_numeric_dtype(df[c0]) and pd.api.types.is_numeric_dtype(df[c1]):
        return None
    return None


def should_chart(result) -> bool:
    """Return True only when a bar chart would help interpret the result."""
    if isinstance(result, pd.Series):
        if len(result) < 2 or len(result) > MAX_BAR_CATEGORIES:
            return False
        return pd.api.types.is_numeric_dtype(result.dtype)

    if not isinstance(result, pd.DataFrame):
        return False

    rows, cols = result.shape
    if cols != 2 or rows < 2:
        return False
    if rows > MAX_BAR_CATEGORIES:
        return False
    if _is_square_numeric_matrix(result):
        return False
    if pd.api.types.is_numeric_dtype(result.iloc[:, 0]) and pd.api.types.is_numeric_dtype(
        result.iloc[:, 1]
    ):
        if rows <= 5:
            return False
    return _pick_xy_columns(result) is not None


def try_make_chart(result):
    if not should_chart(result):
        return None

    if isinstance(result, pd.Series):
        result = result.reset_index()
        if len(result.columns) == 2:
            result.columns = ["category", "value"]
        else:
            return None

    xy = _pick_xy_columns(result)
    if not xy:
        return None

    x_col, y_col = xy
    plot_df = result.head(MAX_BAR_CATEGORIES)
    fig = px.bar(
        plot_df,
        x=x_col,
        y=y_col,
        color=y_col,
        color_continuous_scale=SEQUENTIAL_SCALE,
    )
    _apply_plotly_theme(fig)
    return fig
